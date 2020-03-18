from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, zeros_
from torch import Tensor
from torch.nn import Parameter

__all__ = [
    "LSTMCell",
]


class LSTMCell(nn.Module):
    def __init__(self, i2h_shape: List[int], h2h_shape: List[int]):
        super(LSTMCell, self).__init__()
        # learnable paramters for input gate.
        self.Wi = Parameter(torch.empty(i2h_shape))
        self.Ui = Parameter(torch.empty(h2h_shape))
        self.bi = Parameter(torch.ones(h2h_shape[1]))

        # learnable paramters for forget gate.
        self.Wf = Parameter(torch.empty(i2h_shape))
        self.Uf = Parameter(torch.empty(h2h_shape))
        self.bf = Parameter(h2h_shape[1])

        # learnable paramters for cell candidate.
        self.Wg = Parameter(torch.empty(i2h_shape))
        self.Ug = Parameter(torch.empty(h2h_shape))
        self.bg = Parameter(h2h_shape[1])

        # learnable paramters for output gate.
        self.Wo = Parameter(torch.empty(i2h_shape))
        self.Uo = Parameter(torch.empty(h2h_shape))
        self.bo = Parameter(h2h_shape[1])

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                xavier_normal_(p.data)
            else:
                zeros_(p.data)

    def forward(self, input: Tensor,
                state_prev: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward computation of a LSTM Cell.
        Args:
            input, Tensor, input to current time step with a shape of
                [batch_size, input_size].
            state_prev, Tuple, hidden states of previous time step.
        Returns:
            Hidden states of current time step.
        """

        h_prev, c_prev = state_prev

        ig: Tensor = torch.sigmoid(
            input.mm(self.Wi) + h_prev.mm(self.Ui) + self.bi)

        fg: Tensor = torch.sigmoid(
            input.mm(self.Wf) + h_prev.mm(self.Uf) + self.bf)

        c_candidate: Tensor = torch.tanh(
            input.mm(self.Wg) + h_prev.mm(self.Ug) + self.bg)

        og: Tensor = torch.sigmoid(
            input.mm(self.Wo) + h_prev.mm(self.Uo) + self.bo)

        c = fg.mul(c_prev) + ig.mul(c_candidate)
        h = og.mul(torch.tanh(c))
        return h, c
