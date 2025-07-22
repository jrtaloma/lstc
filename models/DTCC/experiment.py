import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from sklearn.cluster import KMeans
from utils.metrics import evaluation


def train_epoch(args, model, model_augmented, train_loader, test_loader, criterion_rec, criterion_kmeans, criterion_kmeans_augmented, criterion_instance_contrastive, criterion_cluster_contrastive, optimizer, epoch, device):
    model.train()
    model_augmented.train()
    train_loss = []
    train_loss_rec = []
    train_loss_k_means = []
    train_loss_instance_contrastive = []
    train_loss_cluster_contrastive = []

    ### Training on the train set
    for inputs, _, _ in tqdm(train_loader, total=len(train_loader), leave=False):
        inputs = inputs.unsqueeze(-1).to(device)
        inputs_augmented = inputs + torch.empty_like(inputs).normal_(mean=0.0, std=0.3) # random jitter

        optimizer.zero_grad()
        x_rec, z = model(inputs)
        x_rec_augmented, z_augmented = model_augmented(inputs_augmented)

        loss_rec = criterion_rec(x_rec, inputs) + criterion_rec(x_rec_augmented, inputs_augmented)
        loss_k_means = 0.5 * (criterion_kmeans(z) + criterion_kmeans_augmented(z_augmented))
        loss_instance_contrastive = criterion_instance_contrastive(z, z_augmented, args.temperature)
        loss_cluster_contrastive = criterion_cluster_contrastive(criterion_kmeans.F, criterion_kmeans_augmented.F, args.temperature)
        loss_total = loss_rec + loss_k_means + loss_instance_contrastive + loss_cluster_contrastive
        loss_total.backward()
        optimizer.step()
        train_loss.append(loss_total.detach())
        train_loss_rec.append(loss_rec.detach())
        train_loss_k_means.append(loss_k_means.detach())
        train_loss_instance_contrastive.append(loss_instance_contrastive.detach())
        train_loss_cluster_contrastive.append(loss_cluster_contrastive.detach())

        wandb.log({
            'Loss': loss_total,
            'MSE Loss': loss_rec,
            'K-means Loss': loss_k_means,
            'Instance Loss': loss_instance_contrastive,
            'Cluster Loss': loss_cluster_contrastive
        })

        # Updating cluster indicator matrix
        if epoch % args.alter_iter == 0 and epoch != 0:
            criterion_kmeans.update_kmeans_f(z.detach().cpu().numpy())
            criterion_kmeans_augmented.update_kmeans_f(z_augmented.detach().cpu().numpy())

    ### Evaluating on the test set
    model.eval()
    all_z, all_gt = [], []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.unsqueeze(-1).to(device)
            _, z = model(inputs)
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

    train_loss = torch.stack(train_loss).mean().item()
    train_loss_rec = torch.stack(train_loss_rec).mean().item()
    train_loss_k_means = torch.stack(train_loss_k_means).mean().item()
    train_loss_instance_contrastive = torch.stack(train_loss_instance_contrastive).mean().item()
    train_loss_cluster_contrastive = torch.stack(train_loss_cluster_contrastive).mean().item()

    print(
        f'Epoch: {epoch+1}/{args.epochs}',
        'Loss: %.4f' % train_loss,
        'MSE Loss: %.4f' % train_loss_rec,
        'K-means Loss: %.4f' % train_loss_k_means,
        'Instance Loss: %.4f' % train_loss_instance_contrastive,
        'Cluster Loss: %.4f' % train_loss_cluster_contrastive,
        'RI score: %.4f' % ri,
        'ARI score: %.4f' % ari,
        'NMI score: %.4f' % nmi,
        'Silhouette score: %.4f' % silhouette
    )

    wandb.log({
        'Epoch': epoch+1,
        'Loss (epoch)': train_loss,
        'MSE Loss (epoch)': train_loss_rec,
        'K-means Loss (epoch)': train_loss_k_means,
        'Instance Loss (epoch)': train_loss_instance_contrastive,
        'Cluster Loss (epoch)': train_loss_cluster_contrastive,
        'RI score': ri,
        'ARI score': ari,
        'NMI score': nmi,
        'Silhouette score': silhouette
    })

    return all_z, all_preds, all_gt, ri, ari, nmi, silhouette


def train(args, model, model_augmented, train_loader, test_loader, criterion_rec, criterion_kmeans, criterion_kmeans_augmented, criterion_instance_contrastive, criterion_cluster_contrastive, optimizer, path_ckpt, device):
    wandb.init(project='Concrete Dense Network for Long-Sequence Time Series Clustering', config=args)

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

    wandb.finish()

    return epoch+1, preds, ri, ari, nmi, silhouette