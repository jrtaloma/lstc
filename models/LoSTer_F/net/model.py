import torch
import torch.nn as nn
from .revin import RevIN


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


class LoSTer_F(nn.Module):
    def __init__(self,
                 n_steps,
                 n_steps_output,
                 hidden_size,
                 n_encoder_layers,
                 n_decoder_layers,
                 dropout,
                 use_revin,
                 use_residual):
        super(LoSTer_F, self).__init__()
        self.model = AE(n_steps, n_steps_output, hidden_size, n_encoder_layers, n_decoder_layers, dropout, use_revin, use_residual)

    def forward(self, x):
        x_rec, z = self.model(x)
        return x_rec, z
