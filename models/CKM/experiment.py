import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from utils.metrics import evaluation
from utils.gumbel import softmax_logits


def train_epoch(args, model, train_loader, test_loader, criterion_rec, criterion_kmeans, optimizer, scheduler, epoch, device):
    model.train()
    train_loss = []
    train_loss_rec = []
    train_loss_k_means = []

    temperature = max(args.temperature * args.beta**epoch, 0.01)

    ### Training on the train set
    for inputs, _, _ in tqdm(train_loader, total=len(train_loader), leave=False):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        x_rec, z = model(inputs)
        cluster_logits = softmax_logits(z, criterion_kmeans.centroids.to(device))
        loss_rec = criterion_rec(x_rec, inputs)
        loss_k_means = criterion_kmeans(z, cluster_logits, temperature)
        loss_total = loss_rec + args.alpha * loss_k_means
        loss_total.backward()
        optimizer.step()
        train_loss.append(loss_total.detach())
        train_loss_rec.append(loss_rec.detach())
        train_loss_k_means.append(loss_k_means.detach())

        wandb.log({
            'Loss': loss_total,
            'MSE Loss': loss_rec,
            'K-means Loss': loss_k_means
        })

    scheduler.step()

    ### Evaluating on the test set
    model.eval()
    all_z, all_preds, all_gt = [], [], []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            _, z = model(inputs)
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

    train_loss = torch.stack(train_loss).mean().item()
    train_loss_rec = torch.stack(train_loss_rec).mean().item()
    train_loss_k_means = torch.stack(train_loss_k_means).mean().item()

    print(
        f'Epoch: {epoch+1}/{args.epochs}',
        'Loss: %.4f' % train_loss,
        'MSE Loss: %.4f' % train_loss_rec,
        'K-means Loss: %.4f' % train_loss_k_means,
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
        'RI score': ri,
        'ARI score': ari,
        'NMI score': nmi,
        'Silhouette score': silhouette
    })

    return all_z, all_preds, all_gt, ri, ari, nmi, silhouette


def train(args, model, train_loader, test_loader, criterion_rec, criterion_kmeans, optimizer, scheduler, path_ckpt, device):
    wandb.init(project='Concrete Dense Network for Long-Sequence Time Series Clustering', config=args)

    best_ri_score = -np.inf
    print('Training full model ...')

    preds_last = []
    for epoch in tqdm(range(args.epochs), total=args.epochs, leave=False):
        _, preds, _, ri, ari, nmi, silhouette = train_epoch(args, model, train_loader, test_loader, criterion_rec, criterion_kmeans, optimizer, scheduler, epoch, device)
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