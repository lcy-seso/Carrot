from math import sqrt

import torch

from torch.nn import Module, Linear, Embedding
from torch.nn.init import uniform_
from torch import tanh, sigmoid


class LSTM(Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size(int): input dimension
            hidden_size(int): hidden dimension
            output_size(int): output dimension
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.register_buffer('hidden', torch.Tensor())
        self.register_buffer('cell', torch.Tensor())

        self.gate = Linear(input_size + hidden_size, hidden_size)
        self.output = Linear(hidden_size, output_size)

        stdv = 1.0 / sqrt(self.hidden_size)

        for weight in self.parameters():
            uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor):
        # shape of `input` is (seq_len, batch_size, input_size)
        batch_size = input.size(1)
        # shape of `hidden` is (batch_size, hidden_size)
        self.hidden = self.hidden.new_zeros((batch_size, self.hidden_size))
        # shape of `cell` is (batch_size, hidden_size)
        self.cell = self.cell.new_zeros((batch_size, self.hidden_size))
        output = []
        # `x_t` is input data for each time_step. It has a shape of
        # (batch_size, input_size)
        # TODO(hongyu): Make RNNCell configurable.
        for x_t in input:
            combined = torch.cat((x_t, self.hidden), 1)
            f_gate = self.gate(combined)
            i_gate = self.gate(combined)
            o_gate = self.gate(combined)
            f_gate = sigmoid(f_gate)
            i_gate = sigmoid(i_gate)
            o_gate = sigmoid(o_gate)
            self.cell = torch.add(
                torch.mul(self.cell, f_gate),
                torch.mul(tanh(self.gate(combined)), i_gate))
            self.hidden = torch.mul(tanh(self.cell), o_gate)
            output.append(self.output(self.hidden))

        output = torch.stack(output, dim=0)
        return output, self.hidden


class RNNModel(Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid):
        super(RNNModel, self).__init__()
        self.encoder = Embedding(ntoken, ninp)
        self.rnn = LSTM(ninp, nhid, nhid)
        self.decoder = Linear(nhid, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        # shape of `input` is (seq_len, batch_size)
        emb = self.encoder(input)
        # shape of `emb` is (seq_len, batch_size, embedding_size)
        output, hidden = self.rnn(emb)
        # shape of `output` is (seq_len, batch_size, output_size)
        decoded = self.decoder(output)
        return decoded, hidden
