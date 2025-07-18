import math
import torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from net.layers.AMS import AMS
from net.layers.Layer import WeightGenerator, CustomLinear
from net.layers.RevIN import RevIN
from functools import reduce
from operator import mul
from tqdm import tqdm


# Autoencoder pretraining
def pretrain_ae(args, model, train_loader, optimizer, path_pretrain, device=torch.device('cuda')):
    criterion = nn.MSELoss()
    for epoch in tqdm(range(args.epochs_pretrain), total=args.epochs_pretrain, leave=False):
        total_loss = []
        for inputs, _, _ in tqdm(train_loader, total=len(train_loader), leave=False):
            inputs = inputs.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            x_rec, _, balance_loss = model(inputs)
            loss = criterion(x_rec, inputs) + balance_loss
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
            x_rec_augmented, _, balance_loss_augmented = model_augmented(inputs_augmented)
            loss = criterion(x_rec_augmented, inputs_augmented) + balance_loss_augmented
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
            _, z, _ = model(inputs)
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
            _, z_augmented, _ = model_augmented(inputs_augmented)
            all_z_augmented.append(z_augmented.detach().cpu())
    all_z_augmented = torch.cat(all_z_augmented, dim=0)
    return all_z_augmented


class PathFormer(nn.Module):
    def __init__(self,
                 num_layers,
                 seq_len,
                 k,
                 num_experts_list,
                 patch_size_list,
                 d_model,
                 d_ff,
                 residual_connection,
                 revin,
                 device):
        super(PathFormer, self).__init__()
        self.num_layers = num_layers
        self.num_nodes = 1
        self.seq_len = seq_len
        self.k = k
        self.num_experts_list = num_experts_list
        self.patch_size_list = np.array(patch_size_list).reshape(num_layers, -1).tolist()
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_connection = residual_connection
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = device

        for num in range(self.num_layers):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.seq_len)
        )

    def forward(self, x):

        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))

        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        embedding = out.squeeze(1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, embedding, balance_loss