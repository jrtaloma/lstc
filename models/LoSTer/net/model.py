import torch
import torch.nn as nn
from .revin import RevIN
from tqdm import tqdm


# Autoencoder pretraining
def pretrain_ae(args, model, train_loader, optimizer, path_pretrain, device=torch.device('cuda')):
    criterion = nn.MSELoss()
    for epoch in tqdm(range(args.epochs_pretrain), total=args.epochs_pretrain, leave=False):
        total_loss = []
        for inputs, _, _ in tqdm(train_loader, total=len(train_loader), leave=False):
            inputs = inputs.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            x_rec, _ = model(inputs)
            loss = criterion(x_rec, inputs)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach())
        print(
            'Pretraining autoencoder loss for epoch {}: {}'.format(
                epoch+1, torch.stack(total_loss).mean().item()
            )
        )
    torch.save(model.state_dict(), path_pretrain)
    print('Pretrained model saved to: {}'.format(path_pretrain))


# Autoencoder pretraining (augmented view)
def pretrain_ae_augmented(args, model_augmented, train_loader, optimizer, path_pretrain_augmented, device=torch.device('cuda')):
    criterion = nn.MSELoss()
    for epoch in tqdm(range(args.epochs_pretrain), total=args.epochs_pretrain, leave=False):
        total_loss = []
        for _, inputs_augmented, _ in tqdm(train_loader, total=len(train_loader), leave=False):
            inputs_augmented = inputs_augmented.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            x_rec_augmented, _ = model_augmented(inputs_augmented)
            loss = criterion(x_rec_augmented, inputs_augmented)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach())
        print(
            'Pretraining autoencoder loss for epoch {}: {}'.format(
                epoch+1, torch.stack(total_loss).mean().item()
            )
        )
    torch.save(model_augmented.state_dict(), path_pretrain_augmented)
    print('Pretrained augmented model saved to: {}'.format(path_pretrain_augmented))


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


# Autoencoder set prediction (augmented view)
def predict_augmented(model_augmented, loader, device=torch.device('cuda')):
    model_augmented.eval()
    all_z_augmented = []
    with torch.no_grad():
        for _, inputs_augmented, _ in loader:
            inputs_augmented = inputs_augmented.unsqueeze(-1).to(device)
            _, z_augmented = model_augmented(inputs_augmented)
            all_z_augmented.append(z_augmented.detach().cpu())
    all_z_augmented = torch.cat(all_z_augmented, dim=0)
    return all_z_augmented


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


class LoSTer(nn.Module):
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
                 path_pretrain='',
                 device=torch.device('cuda')):
        super(LoSTer, self).__init__()
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
        return x_rec, z
