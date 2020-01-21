from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_ as init
from torch import Tensor

__all__ = [
    "LSTMCell",
]


class LSTMCell(nn.Module):
    def __init__(self, i2h_shape: List[int], h2h_shape: List[int]):
        super(LSTMCell, self).__init__()
        # learnable paramters for input gate.
        self.Wi = init(nn.Parameter(torch.empty(i2h_shape)))
        self.Ui = init(nn.Parameter(torch.empty(h2h_shape)))
        self.bi = nn.Parameter(torch.ones(h2h_shape[1]))

        # learnable paramters for forget gate.
        self.Wf = init(nn.Parameter(torch.empty(i2h_shape)))
        self.Uf = init(nn.Parameter(torch.empty(h2h_shape)))
        self.bf = nn.Parameter(torch.ones(h2h_shape[1]))

        # learnable paramters for cell candidate.
        self.Wg = init(nn.Parameter(torch.empty(i2h_shape)))
        self.Ug = init(nn.Parameter(torch.empty(h2h_shape)))
        self.bg = nn.Parameter(torch.ones(h2h_shape[1]))

        # learnable paramters for output gate.
        self.Wo = init(nn.Parameter(torch.empty(i2h_shape)))
        self.Uo = init(nn.Parameter(torch.empty(h2h_shape)))
        self.bo = nn.Parameter(torch.ones(h2h_shape[1]))

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
