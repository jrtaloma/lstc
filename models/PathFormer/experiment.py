import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from utils.metrics import evaluation
from utils.gumbel import softmax_logits


def train_epoch(args, model, model_augmented, train_loader, test_loader, criterion_rec, criterion_kmeans, criterion_kmeans_augmented, criterion_instance_contrastive, criterion_cluster_contrastive, optimizer, epoch, device):
    model.train()
    model_augmented.train()
    train_loss = []
    train_loss_rec = []
    train_loss_k_means = []
    train_loss_instance_contrastive = []
    train_loss_cluster_contrastive = []

    temperature = max(args.temperature * args.beta**epoch, 0.01)

    ### Training on the train set
    for inputs, inputs_augmented, _ in tqdm(train_loader, total=len(train_loader), leave=False):
        inputs = inputs.unsqueeze(-1).to(device)
        inputs_augmented = inputs_augmented.unsqueeze(-1).to(device)

        optimizer.zero_grad()
        x_rec, z, balance_loss = model(inputs)
        x_rec_augmented, z_augmented, balance_loss_augmented = model_augmented(inputs_augmented)
        cluster_logits = softmax_logits(z, criterion_kmeans.centroids.to(device))
        cluster_logits_augmented = softmax_logits(z_augmented, criterion_kmeans_augmented.centroids.to(device))
        one_hot = F.gumbel_softmax(cluster_logits, tau=temperature, hard=True)
        one_hot_augmented = F.gumbel_softmax(cluster_logits_augmented, tau=temperature, hard=True)

        loss_rec = criterion_rec(x_rec, inputs) + criterion_rec(x_rec_augmented, inputs_augmented)
        loss_k_means = 0.5 * (criterion_kmeans(z, one_hot) + criterion_kmeans_augmented(z_augmented, one_hot_augmented))
        loss_instance_contrastive = criterion_instance_contrastive(z, z_augmented)
        loss_cluster_contrastive = criterion_cluster_contrastive(one_hot, one_hot_augmented)
        loss_total = loss_rec + args.alpha * loss_k_means + loss_instance_contrastive + loss_cluster_contrastive + balance_loss + balance_loss_augmented
        loss_total.backward()
        optimizer.step()
        train_loss.append(loss_total.detach())
        train_loss_rec.append(loss_rec.detach())
        train_loss_k_means.append(loss_k_means.detach())
        train_loss_instance_contrastive.append(loss_instance_contrastive.detach())
        train_loss_cluster_contrastive.append(loss_cluster_contrastive.detach())

    ### Evaluating on the test set
    model.eval()
    all_z, all_preds, all_gt = [], [], []
    with torch.no_grad():
        for inputs, _, labels in test_loader:
            inputs = inputs.unsqueeze(-1).to(device)
            _, z, _ = model(inputs)
            cluster_logits = softmax_logits(z, criterion_kmeans.centroids.detach().to(device))
            preds = torch.argmax(cluster_logits, dim=1)
            all_z.append(z.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_gt.append(labels.detach().cpu())
    all_z = torch.cat(all_z, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_gt = torch.cat(all_gt, dim=0).numpy()
    ri, ari, nmi = evaluation(all_preds, all_gt)
    try:
        silhouette = silhouette_score(all_z, all_preds)
    except:
        silhouette = -np.inf

    print(
        f'Epoch: {epoch+1}/{args.epochs}',
        'Loss: %.4f' % torch.stack(train_loss).mean().item(),
        'MSE Loss: %.4f' % torch.stack(train_loss_rec).mean().item(),
        'K-means Loss: %.4f' % torch.stack(train_loss_k_means).mean().item(),
        'Instance Loss: %.4f' % torch.stack(train_loss_instance_contrastive).mean().item(),
        'Cluster Loss: %.4f' % torch.stack(train_loss_cluster_contrastive).mean().item(),
        'RI score: %.4f' % ri,
        'ARI score: %.4f' % ari,
        'NMI score: %.4f' % nmi,
        'Silhouette score: %.4f' % silhouette
    )

    return all_z, all_preds, all_gt, ri, ari, nmi, silhouette


def train(args, model, model_augmented, train_loader, test_loader, criterion_rec, criterion_kmeans, criterion_kmeans_augmented, criterion_instance_contrastive, criterion_cluster_contrastive, optimizer, path_ckpt, device):
    best_ri_score = -np.inf
    print('Training full model ...')

    preds_last = []
    for epoch in tqdm(range(args.epochs), total=args.epochs, leave=False):
        _, preds, _, ri, ari, nmi, silhouette = train_epoch(args, model, model_augmented, train_loader, test_loader, criterion_rec, criterion_kmeans, criterion_kmeans_augmented, criterion_instance_contrastive, criterion_cluster_contrastive, optimizer, epoch, device)
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
