from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, zeros_
from torch.nn import Module

__all__ = [
    'MogLSTMCell',
]


class MogLSTMCell(Module):
    """Cell function of the Mogrifier LSTM."""

    def __init__(self, input_size: int, hidden_size: int, mog_iterations: int):
        """
        Args:
            input_size, int, The number of neurons in RNN's input layer.
            hidden_size, int, The number of neurons in RNN's hidden layer.
            mog_iterations, int, The number of rounds for mogrifications.
        """
        super(MogLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mog_iterations = mog_iterations

        self.Wih = Parameter(Tensor(input_size, hidden_size * 4))
        self.Whh = Parameter(Tensor(hidden_size, hidden_size * 4))
        self.bih = Parameter(Tensor(hidden_size * 4))
        self.bhh = Parameter(Tensor(hidden_size * 4))

        # Mogrifiers
        self.Q = Parameter(Tensor(hidden_size, input_size))
        self.R = Parameter(Tensor(input_size, hidden_size))

        self.init_weights()
        self.register_buffer('init_state', torch.zeros(1, self.hidden_size))

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                xavier_normal_(p.data)
            else:
                zeros_(p.data)

    def mogrify(self, xt, ht):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht

    def forward(self,
                xt: Tensor,
                init_states: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if init_states is None:
            ht = self.init_state
            ct = self.init_state
        else:
            ht, ct = init_states

        xt, ht = self.mogrify(xt, ht)  # mogrification
        gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # The standard LSTM Cell.
        ft = torch.sigmoid(forgetgate)
        it = torch.sigmoid(ingate)
        ct_candidate = torch.tanh(cellgate)
        ot = torch.sigmoid(outgate)

        ct = (ft * ct) + (it * ct_candidate)
        ht = ot * torch.tanh(ct)

        return ht, (ht, ct)
