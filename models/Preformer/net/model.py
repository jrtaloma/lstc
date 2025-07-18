import torch
import torch.nn as nn
from net.layers.Embed import DataEmbedding_wo_pos
from net.layers.SelfAttention_Family import SegmentCorrelation2, SegmentCorrelation3, MultiScaleAttentionLayer
from net.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
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


class Preformer(nn.Module):
    def __init__(self,
                 seq_len,
                 output_attention,
                 moving_avg,
                 d_model,
                 n_heads,
                 dropout,
                 factor,
                 d_ff,
                 activation,
                 e_layers,
                 d_layers,
                 ):
        super(Preformer, self).__init__()
        self.seq_len = seq_len
        self.output_attention = output_attention
        self.moving_avg = moving_avg
        self.enc_in = 1
        self.dec_in = 1
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.factor = factor
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.c_out = 1

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiScaleAttentionLayer(
                        SegmentCorrelation2(False, self.factor, attention_dropout=self.dropout,
                                            output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiScaleAttentionLayer(
                        SegmentCorrelation2(True, self.factor, attention_dropout=self.dropout,
                                            output_attention=False),
                        self.d_model, self.n_heads),
                    MultiScaleAttentionLayer(
                        SegmentCorrelation3(False, self.factor, attention_dropout=self.dropout,
                                            output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        seasonal_init, trend_init = self.decomp(x_enc)

        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        embedding = enc_out[:,-1]
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.seq_len:, :], embedding, attns
        else:
            return dec_out[:, -self.seq_len:, :], embedding  # [B, L, D], [B, D]