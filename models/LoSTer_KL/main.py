import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from utils.config import get_arguments
from utils.random_seed import random_seed
from data.load_data import get_loader
from net.model import AE, LoSTer_KL, pretrain_ae, predict
from experiment import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    path_pretrain = os.path.join(
        path_ckpts, 'autoencoder.pth'
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
    args.n_input = train_dataset.n_input

    # Autoencoder
    model = AE(n_steps=args.n_input, n_steps_output=args.n_input, hidden_size=args.hidden_size, n_encoder_layers=args.n_encoder_layers, n_decoder_layers=args.n_decoder_layers, dropout=args.dropout, use_revin=args.use_revin, use_residual=False)
    model = model.to(device)

    # Pretraining autoencoder or loading pretrained
    if args.epochs_pretrain == 0:
        model.load_state_dict(torch.load(path_pretrain, map_location=device))
    if args.epochs_pretrain > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pretrain_ae(args, model, train_loader, optimizer, path_pretrain, device)

    # Initializing full model with centroids
    all_z = predict(model, test_loader, device=torch.device('cuda'))
    kmeans = KMeans(n_clusters=args.n_clusters, init='k-means++', random_state=args.seed)
    kmeans.fit(all_z.data.cpu().numpy())
    model = LoSTer_KL(n_steps=args.n_input, n_steps_output=args.n_input, hidden_size=args.hidden_size, n_encoder_layers=args.n_encoder_layers, n_decoder_layers=args.n_decoder_layers, dropout=args.dropout, use_revin=args.use_revin, use_residual=False, centroids=kmeans.cluster_centers_, alpha=args.alpha, path_pretrain=path_pretrain, device=device)
    model = model.to(device)

    criterion_rec = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Training & testing
    data = torch.tensor(train_dataset.time_series, device=device).unsqueeze(-1).data
    epoch, preds, probs, ri, ari, nmi, silhouette = train(args, model, data, train_loader, test_loader, criterion_rec, optimizer, scheduler, path_ckpt, device)
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