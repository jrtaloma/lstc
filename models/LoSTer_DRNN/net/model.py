import numpy as np
import torch
import torch.nn as nn
from .drnn import DRNN, SingleRNN
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


class LoSTer_DRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 dilations,
                 n_steps,
                 cell_type,
                 centroids,
                 path_pretrain='',
                 device=torch.device('cuda')):
        super(LoSTer_DRNN, self).__init__()
        assert len(hidden_sizes) == len(dilations)
        self.path_pretrain = path_pretrain
        self.device = device
        self.ae = DilatedRNN(input_size, hidden_sizes, dilations, n_steps, cell_type)
        # Loading pretrained AE
        if self.path_pretrain != '':
            print(f'Loading pretrained AE from: {self.path_pretrain}')
            self.ae.load_state_dict(torch.load(self.path_pretrain, map_location=self.device))
        # Initializing centroids
        self.centroids = nn.Parameter(torch.tensor(centroids))

    def forward(self, x):
        x_rec, z = self.ae(x)
        return x_rec, z
