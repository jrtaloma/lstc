import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def compute_similarity(z, centroids):
    """
    Function that compute squared distance between a latent vector z and the clusters centroids.
    z shape : (batch_size, hidden_size)
    centroids shape : (n_clusters, hidden_size)
    output : (batch_size, n_clusters)
    """
    n_clusters, hidden_size = centroids.shape[0], centroids.shape[1]
    batch_size = z.shape[0]
    z = z.expand((n_clusters, batch_size, hidden_size))
    dist_sq = torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2)
    return torch.transpose(dist_sq, 0, 1)


# Autoencoder pretraining
def pretrain_ae(args, model, train_loader, optimizer, path_pretrain, device=torch.device('cuda'), use_multiple_gpus=False):
    criterion = nn.MSELoss()
    for epoch in tqdm(range(args.epochs_pretrain), total=args.epochs_pretrain, leave=False):
        total_loss = 0
        for batch_idx, (inputs, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            x_rec, _ = model(inputs)
            loss = criterion(x_rec, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            'Pretraining autoencoder loss for epoch {}: {}'.format(
                epoch+1, total_loss / (batch_idx + 1)
            )
        )
    if use_multiple_gpus:
        torch.save(model.module.state_dict(), path_pretrain)
    else:
        torch.save(model.state_dict(), path_pretrain)
    print('Pretrained model saved to: {}'.format(path_pretrain))


# Autoencoder set prediction
def predict(model, loader, device=torch.device('cuda')):
    model.eval()
    all_z = []
    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(device)
            _, z = model(inputs)
            all_z.append(z.detach().cpu())
    all_z = torch.cat(all_z, dim=0)
    return all_z


# https://github.com/dawnranger/IDEC-pytorch/blob/master/idec.py
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)

        self.z_layer = nn.Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)

        self.x_rec_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_rec = self.x_rec_layer(dec_h3)

        return x_rec, z


class IDEC(nn.Module):
    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 centroids,
                 alpha=1.0,
                 path_pretrain='',
                 device=torch.device('cuda')):
        super(IDEC, self).__init__()
        self.alpha = alpha
        self.path_pretrain = path_pretrain
        self.device = device
        self.ae = AE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z)
        # Loading pretrained AE
        if self.path_pretrain != '':
            print(f'Loading pretrained AE from: {self.path_pretrain}')
            self.ae.load_state_dict(torch.load(self.path_pretrain, map_location=self.device))
        # Initializing centroids
        self.centroids = nn.Parameter(torch.tensor(centroids))

    def forward(self, x):
        x_rec, z = self.ae(x)
        similarity = compute_similarity(z, self.centroids)

        ## Q (batch_size, n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha)), -(self.alpha + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        return x_rec, z, Q
