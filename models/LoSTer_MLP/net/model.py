import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN
from tqdm import tqdm


# Autoencoder pretraining
def pretrain_ae(args, model, train_loader, optimizer, path_pretrain, device=torch.device('cuda')):
    criterion = nn.MSELoss()
    for epoch in tqdm(range(args.epochs_pretrain), total=args.epochs_pretrain, leave=False):
        total_loss = []
        for inputs, _, _ in tqdm(train_loader, total=len(train_loader), leave=False):
            inputs = inputs.to(device)
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
            inputs_augmented = inputs_augmented.to(device)
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
            inputs = inputs.to(device)
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
            inputs_augmented = inputs_augmented.to(device)
            _, z_augmented = model_augmented(inputs_augmented)
            all_z_augmented.append(z_augmented.detach().cpu())
    all_z_augmented = torch.cat(all_z_augmented, dim=0)
    return all_z_augmented


# https://github.com/dawnranger/IDEC-pytorch/blob/master/idec.py
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_output, n_z, use_revin):
        super(AE, self).__init__()
        self.use_revin = use_revin
        if self.use_revin: self.revin_layer = RevIN(num_features=1, affine=True, subtract_last=False)

        # encoder
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)

        self.z_layer = nn.Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)

        self.x_rec_layer = nn.Linear(n_dec_3, n_output)

    def forward(self, x):
        if self.use_revin:
            x = x.unsqueeze(-1)
            x = self.revin_layer(x, mode='norm')
            x = x.squeeze(-1)

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

        if self.use_revin:
            x_rec = x_rec.unsqueeze(-1)
            x_rec = self.revin_layer(x_rec, mode='denorm')
            x_rec = x_rec.squeeze(-1)

        return x_rec, z


class LoSTer_MLP(nn.Module):
    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_output,
                 n_z,
                 use_revin,
                 centroids,
                 path_pretrain='',
                 device=torch.device('cuda')):
        super(LoSTer_MLP, self).__init__()
        self.path_pretrain = path_pretrain
        self.device = device
        self.ae = AE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_output, n_z, use_revin)
        # Loading pretrained AE
        if self.path_pretrain != '':
            print(f'Loading pretrained AE from: {self.path_pretrain}')
            self.ae.load_state_dict(torch.load(self.path_pretrain, map_location=self.device))
        # Initializing centroids
        self.centroids = nn.Parameter(torch.tensor(centroids))

    def forward(self, x):
        x_rec, z = self.ae(x)
        return x_rec, z
