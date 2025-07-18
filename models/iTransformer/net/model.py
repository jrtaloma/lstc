import torch
import torch.nn as nn
from net.layers.Transformer_EncDec import Encoder, EncoderLayer
from net.layers.SelfAttention_Family import FullAttention, AttentionLayer
from net.layers.Embed import DataEmbedding_inverted
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


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self,
                 seq_len,
                 d_model,
                 dropout,
                 factor,
                 n_heads,
                 d_ff,
                 activation,
                 e_layers,
                 output_attention,
                 use_norm
                 ):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.output_attention = output_attention
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projector = nn.Linear(self.d_model, self.seq_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: self.;       S: self.;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc)
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        embedding = enc_out.squeeze(1)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return dec_out, embedding


    def forward(self, x_enc):
        dec_out, embedding = self.forecast(x_enc)
        return dec_out[:, -self.seq_len:, :], embedding  # [B, L, D], [B, D]