"""
https://github.com/zalandoresearch/pytorch-dilated-rnn
"""

import torch
import torch.nn as nn


use_cuda = torch.cuda.is_available()


class DRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dilations, cell_type='LSTM'):
        super(DRNN, self).__init__()

        assert len(hidden_sizes) == len(dilations), f'The number of hidden states and of dilation rates are different'

        self.dilations = dilations
        self.cell_type = cell_type

        layers = []
        if self.cell_type == 'GRU':
            cell = nn.GRU
        elif self.cell_type == 'RNN':
            cell = nn.RNN
        elif self.cell_type == 'LSTM':
            cell = nn.LSTM
        else:
            raise ValueError

        for i in range(len(dilations)):
            if i == 0:
                c = cell(input_size, hidden_sizes[i])
            else:
                c = cell(hidden_sizes[i-1], hidden_sizes[i])
            layers.append(c)
        self.cells = nn.Sequential(*layers)


    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)
        outputs = []
        for cell, dilation in zip(self.cells, self.dilations):
            inputs, _ = self.drnn_layer(cell, inputs, dilation)
            outputs.append(inputs)
        inputs = inputs.transpose(0, 1)
        return inputs, outputs


    def drnn_layer(self, cell, inputs, rate):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size
        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)
        dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)
        return outputs, hidden


    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size):
        if self.cell_type == 'LSTM':
            c, m = self.init_hidden(batch_size * rate, hidden_size)
            hidden = (c.unsqueeze(0), m.unsqueeze(0))
        else:
            hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)
        dilated_outputs, hidden = cell(dilated_inputs, hidden)
        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]


    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved


    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps


    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs


    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden


class SingleRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_steps, cell_type='LSTM'):
        super(SingleRNN, self).__init__()

        if cell_type == 'GRU':
            cell = nn.GRUCell
        elif cell_type == 'RNN':
            cell = nn.RNNCell
        elif cell_type == 'LSTM':
            cell = nn.LSTMCell
        else:
            raise ValueError

        self.input_size = input_size
        self.n_steps = n_steps
        self.cell_type = cell_type
        self.cell = cell(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.ModuleList([nn.Linear(hidden_size, input_size) for _ in range(n_steps)])

    def forward(self, hidden):
        batch_size = hidden.shape[0]
        x = torch.zeros(batch_size, self.input_size, dtype=torch.float32, device=hidden.device)

        hx = hidden
        if self.cell_type == 'LSTM':
            cx = torch.zeros_like(hx)

        outputs = []
        for i in range(self.n_steps):
            if self.cell_type == 'LSTM':
                hx, cx = self.cell(x, (hx, cx))
            else:
                hx = self.cell(x, hx)
            x = self.linear[i](hx)
            outputs.append(x)
        outputs = torch.stack(outputs, dim=1)

        return outputs
