import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN
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


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Dropout(dropout)
        )
        self.residual = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        output = self.block(x) + self.residual(x)
        output = self.layer_norm(output)
        return output


class PredictionResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictionResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.residual = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.block(x) + self.residual(x)
        return output


class TemporalResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TemporalResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.residual = nn.Linear(input_size, 1)

    def forward(self, x):
        output = self.block(x) + self.residual(x)
        return output


class AE(nn.Module):
    def __init__(self, n_steps, n_steps_output, hidden_size, n_encoder_layers, n_decoder_layers, dropout, use_revin, use_residual):
        super(AE, self).__init__()
        self.input_size = n_steps
        self.hidden_size = hidden_size
        self.output_size = n_steps_output
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.use_revin = use_revin
        self.use_residual = use_residual

        if self.use_revin: self.revin_layer = RevIN(num_features=1, affine=True, subtract_last=False)

        encoder_layers = []
        for i in range(self.n_encoder_layers):
            if i == 0:
                residual_block = ResidualBlock(self.input_size, self.hidden_size, self.hidden_size, self.dropout)
            else:
                residual_block = ResidualBlock(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)
            encoder_layers.append(residual_block)
        self.dense_encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self.n_decoder_layers):
            if i == self.n_decoder_layers - 1:
                residual_block = PredictionResidualBlock(self.hidden_size, self.hidden_size, self.output_size)
            else:
                residual_block = ResidualBlock(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)
            decoder_layers.append(residual_block)
        self.dense_decoder = nn.Sequential(*decoder_layers)

        if self.use_residual:
            self.residual = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        # assert x.dim() == 3 and x.size(2) == 1, f'Expected x.shape: [batch_size, n_steps, 1]'
        input = x
        if self.use_revin:
            input = self.revin_layer(input, mode='norm')
        input = input.squeeze(2)
        encoder_state = self.dense_encoder(input)
        decoder_outputs = self.dense_decoder(encoder_state)
        if self.use_residual:
            decoder_outputs = decoder_outputs + self.residual(input)
        decoder_outputs = decoder_outputs.unsqueeze(2)
        if self.use_revin:
            decoder_outputs = self.revin_layer(decoder_outputs, mode='denorm')
        return decoder_outputs, encoder_state


class LoSTer_KL(nn.Module):
    def __init__(self,
                 n_steps,
                 n_steps_output,
                 hidden_size,
                 n_encoder_layers,
                 n_decoder_layers,
                 dropout,
                 use_revin,
                 use_residual,
                 centroids,
                 alpha=1.0,
                 path_pretrain='',
                 device=torch.device('cuda')):
        super(LoSTer_KL, self).__init__()
        self.alpha = alpha
        self.path_pretrain = path_pretrain
        self.device = device
        self.ae = AE(n_steps, n_steps_output, hidden_size, n_encoder_layers, n_decoder_layers, dropout, use_revin, use_residual)
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
