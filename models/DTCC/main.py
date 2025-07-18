import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.config import get_arguments
from utils.random_seed import random_seed
from data.load_data import get_loader
from net.model import DTCC
from utils.losses import KMeansLoss, InstanceContrastiveLoss, ClusterContrastiveLoss
from experiment import train


device = torch.device('cuda')


if __name__ == '__main__':
    ##### Args #####
    parser = get_arguments()
    args = parser.parse_args()
    print(args)

    random_seed(args.seed)
    print(f'Fixing random seed: {args.seed}')

    path_data = os.path.join('./datasets', 'UCRArchive_2018', args.dataset_name, args.experiment_name)
    path_ckpts = os.path.join('./ckpts', 'UCRArchive_2018', args.dataset_name, args.experiment_name)
    path_logs = os.path.join('./logs', 'UCRArchive_2018', args.dataset_name, args.experiment_name)

    os.makedirs(path_ckpts, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)

    path_ckpt = os.path.join(
        path_ckpts, 'model.pth'
    )

    path_log = os.path.join(
        path_logs, 'log.txt'
    )

    path_csv = os.path.join(
        path_logs, f'metrics_{args.model_name}_{args.seed}.csv'
    )

    log_file = open(path_log, 'a+')
    log_file.write(f'\n{str(args)}')

    train_dataset, train_loader = get_loader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    _, test_loader = get_loader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    args.n_clusters = train_dataset.n_clusters
    args.n_steps = train_dataset.n_input
    args.input_size = 1

    # Autoencoders
    model = DTCC(input_size=args.input_size, hidden_sizes=args.hidden_sizes, dilations=args.dilations, n_steps=args.n_steps, cell_type=args.cell_type)
    model = model.to(device)
    model_augmented = DTCC(input_size=args.input_size, hidden_sizes=args.hidden_sizes, dilations=args.dilations, n_steps=args.n_steps, cell_type=args.cell_type)
    model_augmented = model_augmented.to(device)

    criterion_rec = nn.MSELoss()
    criterion_kmeans = KMeansLoss(n_samples=args.batch_size, n_clusters=args.n_clusters, weight=args.weight, device=device)
    criterion_kmeans_augmented = KMeansLoss(n_samples=args.batch_size, n_clusters=args.n_clusters, weight=args.weight, device=device)
    criterion_instance_contrastive = InstanceContrastiveLoss()
    criterion_cluster_contrastive = ClusterContrastiveLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(model_augmented.parameters()), lr=args.lr)

    # Training & testing
    epoch, preds, ri, ari, nmi, silhouette = train(args, model, model_augmented, train_loader, test_loader, criterion_rec, criterion_kmeans, criterion_kmeans_augmented, criterion_instance_contrastive, criterion_cluster_contrastive, optimizer, path_ckpt, device)
    print('Model: {}, Seed: {}, Epoch: {}, RI score (Test): {:.4f}, ARI score (Test): {:.4f}, NMI score (Test): {:.4f}, Silhouette score (Test): {:.4f}'.format(args.model_name, args.seed, epoch, ri, ari, nmi, silhouette))
    log_file.write('\nModel: {}, Seed: {}, Epoch: {}, RI score (Test): {:.4f}, ARI score (Test): {:.4f}, NMI score (Test): {:.4f}, Silhouette score (Test): {:.4f}'.format(args.model_name, args.seed, epoch, ri, ari, nmi, silhouette))

    # Saving metrics on CSV
    metrics = {
        'Dataset': [args.dataset_name],
        'Model': [args.model_name],
        'Seed': [args.seed],
        'Epoch': [epoch],
        'RI': [ri],
        'ARI': [ari],
        'NMI': [nmi],
        'Silhouette': [silhouette]
    }
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics.to_csv(path_csv, index=False)

    # Hard cluster assignments
    print('Hard cluster assignments...')
    log_file.write('\nHard cluster assignments...')
    cluster_assignments = []
    for i in range(args.n_clusters):
        cluster = np.argwhere(preds==i).squeeze()
        cluster_assignments.append(cluster)
        try:
            print(f'{i}: {len(cluster)}')
            log_file.write(f'\n{i}: {len(cluster)}')
        except:
            print(f'{i}: 1')
            log_file.write(f'\n{i}: 1')

    log_file.close()
