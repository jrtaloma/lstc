import numpy as np
import torch
import torch.nn as nn
from .drnn import DRNN, SingleRNN


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


class DTCC(nn.Module):
    def __init__(self, input_size, hidden_sizes, dilations, n_steps, cell_type):
        super().__init__()
        assert len(hidden_sizes) == len(dilations)
        self.model = DilatedRNN(input_size, hidden_sizes, dilations, n_steps, cell_type)

    def forward(self, x):
        x_rec, z = self.model(x)
        return x_rec, z
