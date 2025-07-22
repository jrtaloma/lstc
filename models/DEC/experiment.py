import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from utils.losses import target_distribution, kl_loss_function
from utils.metrics import evaluation


def train_epoch(args, model, train_loader, test_loader, optimizer, scheduler, epoch, P, device):
    model.train()
    train_loss = []

    ### Training on the train set
    for inputs, _, idx in tqdm(train_loader, total=len(train_loader), leave=False):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        x_rec, z, Q = model(inputs)
        loss_total = kl_loss_function(P[idx], Q)
        loss_total.backward()
        optimizer.step()
        train_loss.append(loss_total.detach())

        wandb.log({
            'Loss': loss_total
        })

    scheduler.step()

    ### Evaluating on the test set
    model.eval()
    all_z, all_preds, all_probs, all_gt = [], [], [], []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            _, z, Q = model(inputs)
            preds = torch.max(Q, dim=1)[1]
            all_z.append(z.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_probs.append(Q.detach().cpu())
            all_gt.append(labels.detach().cpu())
    all_z = torch.cat(all_z, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_gt = torch.cat(all_gt, dim=0).numpy()
    ri, ari, nmi = evaluation(all_preds, all_gt)
    try:
        silhouette = silhouette_score(all_z, all_preds)
    except:
        silhouette = -np.inf

    train_loss = torch.stack(train_loss).mean().item()

    print(
        f'Epoch: {epoch+1}/{args.epochs}',
        'Loss: %.4f' % train_loss,
        'RI score: %.4f' % ri,
        'ARI score: %.4f' % ari,
        'NMI score: %.4f' % nmi,
        'Silhouette score: %.4f' % silhouette
    )

    wandb.log({
        'Epoch': epoch+1,
        'Loss (epoch)': train_loss,
        'RI score': ri,
        'ARI score': ari,
        'NMI score': nmi,
        'Silhouette score': silhouette
    })

    return all_z, all_preds, all_probs, all_gt, ri, ari, nmi, silhouette


def train(args, model, data, train_loader, test_loader, optimizer, scheduler, path_ckpt, device):
    wandb.init(project='Concrete Dense Network for Long-Sequence Time Series Clustering', config=args)

    best_ri_score = -np.inf
    print('Training full model ...')

    preds_last = []
    for epoch in tqdm(range(args.epochs), total=args.epochs, leave=False):
        # Update target distribution P
        _, _, Q = model(data)
        P = target_distribution(Q.data)

        _, preds, probs, _, ri, ari, nmi, silhouette = train_epoch(args, model, train_loader, test_loader, optimizer, scheduler, epoch, P, device)
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

    return epoch+1, preds, probs, ri, ari, nmi, silhouette