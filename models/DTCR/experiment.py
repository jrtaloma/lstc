import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from utils.metrics import evaluation


def get_fake_sample(data):
    sample_nums = data.shape[0]
    series_len = data.shape[1]
    mask = torch.ones(sample_nums, series_len, device=data.device)
    rand_list = torch.zeros(sample_nums, series_len, device=data.device)

    fake_position_nums = int(series_len * 0.2)
    fake_position = torch.randint(low=0, high=series_len, size=(sample_nums, fake_position_nums), device=data.device)

    for i in range(fake_position.shape[0]):
        for j in range(fake_position.shape[1]):
            mask[i, fake_position[i, j]] = 0

    for i in range(rand_list.shape[0]):
        count = 0
        for j in range(rand_list.shape[1]):
            if j in fake_position[i]:
                rand_list[i, j] = data[i, fake_position[i, count]]
                count += 1

    fake_data = data * mask[:,:,None] + rand_list[:,:,None] * (1 - mask[:,:,None])

    return fake_data


def train_epoch(args, model, train_loader, test_loader, criterion_rec, criterion_class, criterion_kmeans, optimizer, epoch, device):
    model.train()
    train_loss = 0
    train_loss_rec = 0
    train_loss_k_means = 0
    train_loss_classification = 0

    ### Training on the train set
    for batch_idx, (inputs, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        inputs = inputs.unsqueeze(-1).to(device)
        inputs_fake = get_fake_sample(inputs)
        inputs_real_fake = torch.cat([inputs, inputs_fake], dim=0)
        zeros = torch.zeros(len(inputs), 1, dtype=torch.float32, device=device, requires_grad=False)
        ones = torch.ones(len(inputs), 1, dtype=torch.float32, device=device, requires_grad=False)
        labels_real_fake = torch.cat([ones, zeros], dim=0)

        optimizer.zero_grad()
        x_rec_real_fake, z_real_fake, real_fake_logits = model(inputs_real_fake)
        z = z_real_fake[:len(inputs)]
        x_rec = x_rec_real_fake[:len(inputs)]

        loss_rec = criterion_rec(x_rec, inputs)
        loss_classification = criterion_class(real_fake_logits, labels_real_fake)
        loss_k_means = criterion_kmeans(z)
        loss_total = loss_rec + loss_k_means + loss_classification
        loss_total.backward()
        optimizer.step()
        train_loss += loss_total.item()
        train_loss_rec += loss_rec.item()
        train_loss_k_means += loss_k_means.item()
        train_loss_classification += loss_classification.item()

        # Updating cluster indicator matrix
        if epoch % args.alter_iter == 0 and epoch != 0:
            criterion_kmeans.update_kmeans_f(z.detach().cpu().numpy())

    ### Evaluating on the test set
    model.eval()
    all_z, all_gt = [], []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.unsqueeze(-1).to(device)
            _, z, _ = model(inputs)
            all_z.append(z.detach().cpu())
            all_gt.append(labels.detach().cpu())
    all_z = torch.cat(all_z, dim=0).numpy()
    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_preds = KMeans(n_clusters=args.n_clusters, n_init='auto', random_state=args.seed).fit_predict(all_z)
    ri, ari, nmi = evaluation(all_preds, all_gt)
    try:
        silhouette = silhouette_score(all_z, all_preds)
    except:
        silhouette = -np.inf

    print(
        f'Epoch: {epoch+1}/{args.epochs}',
        'Loss: %.4f' % (train_loss / (batch_idx + 1)),
        'MSE Loss: %.4f' % (train_loss_rec / (batch_idx + 1)),
        'K-means Loss: %.4f' % (train_loss_k_means / (batch_idx + 1)),
        'Classification Loss: %.4f' % (train_loss_classification / (batch_idx + 1)),
        'RI score: %.4f' % ri,
        'ARI score: %.4f' % ari,
        'NMI score: %.4f' % nmi,
        'Silhouette score: %.4f' % silhouette
    )

    return all_z, all_preds, all_gt, ri, ari, nmi, silhouette


def train(args, model, train_loader, test_loader, criterion_rec, criterion_class, criterion_kmeans, optimizer, path_ckpt, device):
    best_ri_score = -np.inf
    print('Training full model ...')

    preds_last = []
    for epoch in tqdm(range(args.epochs), total=args.epochs, leave=False):
        _, preds, _, ri, ari, nmi, silhouette = train_epoch(args, model, train_loader, test_loader, criterion_rec, criterion_class, criterion_kmeans, optimizer, epoch, device)
        if ri > best_ri_score:
            best_ri_score = ri
            print('Epoch: {}/{}, Test RI score: {:.4f}.'.format(epoch+1, args.epochs, best_ri_score))
        if epoch > 0:
            delta_label = np.sum(preds != preds_last).astype(np.float32) / preds.shape[0]
            if delta_label < args.tol:
                print('delta_label {}'.format(delta_label), '< tol', args.tol)
                print('Reached tolerance threshold: stopping training')
                break
        preds_last = preds

    torch.save(model.state_dict(), path_ckpt)
    print('Model saved to: {}'.format(path_ckpt))

    return epoch+1, preds, ri, ari, nmi, silhouette