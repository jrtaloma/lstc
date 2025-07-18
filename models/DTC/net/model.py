import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .drnn import DRNN, SingleRNN


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
            inputs = inputs.unsqueeze(-1).to(device)
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
            inputs = inputs.unsqueeze(-1).to(device)
            _, z = model(inputs)
            all_z.append(z.detach().cpu())
    all_z = torch.cat(all_z, dim=0)
    return all_z


class DilatedRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dilations, n_steps, cell_type):
        super().__init__()
        self.encoder_fw = DRNN(input_size, hidden_sizes, dilations, cell_type)
        self.encoder_bw = DRNN(input_size, hidden_sizes, dilations, cell_type)
        hidden_size = np.sum(hidden_sizes) * 2
        self.decoder = SingleRNN(input_size, hidden_size, n_steps, cell_type)
        self.hidden_sizes = hidden_sizes

    def forward(self, x):
        x_reverse = torch.flip(x, [1])
        _, encoder_output_fw = self.encoder_fw(x)
        _, encoder_output_bw = self.encoder_bw(x_reverse)
        #print([_.shape for _ in encoder_output_fw])
        #print([_.shape for _ in encoder_output_bw])

        fw = []
        bw = []
        for i in range(len(self.hidden_sizes)):
            fw.append(encoder_output_fw[i][-1,:,:])
            bw.append(encoder_output_bw[i][-1,:,:])
        encoder_state_fw = torch.cat(fw, dim=1)
        encoder_state_bw = torch.cat(bw, dim=1)
        #print(encoder_state_fw.shape)
        #print(encoder_state_bw.shape)

        # encoder_state has shape [batch_size, sum(hidden_sizes)*2]
        encoder_state = torch.cat([encoder_state_fw, encoder_state_bw], dim=1)
        #print(encoder_state.shape)

        decoder_outputs = self.decoder(encoder_state)
        return decoder_outputs, encoder_state


class DTC(nn.Module):
    def __init__(self, input_size, hidden_sizes, dilations, n_steps, cell_type, centroids, alpha=1.0, path_pretrain='', device=torch.device('cuda')):
        super().__init__()
        assert len(hidden_sizes) == len(dilations)
        self.alpha = alpha
        self.path_pretrain = path_pretrain
        self.device = device
        self.model = DilatedRNN(input_size, hidden_sizes, dilations, n_steps, cell_type)
        # Loading pretrained AE
        if self.path_pretrain != '':
            print(f'Loading pretrained AE from: {self.path_pretrain}')
            self.model.load_state_dict(torch.load(self.path_pretrain, map_location=self.device))
        # Initializing centroids
        self.centroids = nn.Parameter(torch.tensor(centroids))

    def forward(self, x):
        x_rec, z = self.model(x)
        similarity = compute_similarity(z, self.centroids)

        ## Q (batch_size, n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha)), -(self.alpha + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        return x_rec, z, Q
