from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, zeros_
from torch.nn import Module

__all__ = [
    'VanillaRNNCell',
    'VanillaRNNCell_',
]


class VanillaRNNCell(Module):
    """Cell computation of the Vanilla RNN.

    This implementation can be automatically differentiated.
    """

    def __init__(self, input_size, hidden_size):
        super(VanillaRNNCell, self).__init__()
        # learnable paramters
        self.W = Parameter(Tensor(input_size, hidden_size))
        self.U = Parameter(Tensor(hidden_size, hidden_size))
        self.b = Parameter(Tensor(1, hidden_size))

        self.init_weights()
        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                xavier_normal_(p.data)
            else:
                zeros_(p.data)

    def forward(self, input: Tensor,
                h_prev: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input, Tensor, input to current time step with a shape of
                [batch_size, input_size].
            h_prev, Tuple, hidden state of previous time step.

        Returns:
            Hidden states of current time step.
        """

        h_prev = self.init_state if h_prev is None else h_prev
        return torch.tanh(
            torch.mm(input, self.W) + torch.mm(h_prev, self.U) + self.b)


class VanillaRNNCell_(object):
    """Inplace version. Cell computation of a Vanilla RNN.

    NOTE: This implementation cannot be automatically differentiated.
    """

    def __init__(self, input_size, hidden_size, device, grid_dim=1):
        super(VanillaRNNCell_, self).__init__()
        # learnable paramters
        self.W = xavier_normal_(
            torch.empty(input_size, hidden_size, device=device))
        self.U = xavier_normal_(
            torch.empty(hidden_size * grid_dim, hidden_size, device=device))
        self.b = torch.ones(1, hidden_size, device=device)

        self.device = device

    def forward(self, out: Tensor, x: Tensor,
                h_prev: Optional[Tensor] = None) -> Tensor:
        """Cell computation of a Vanilla RNN.
        Args:
            input, Tensor, input to current time step with a shape of
                [batch_size, input_size].
            h_prev, Tuple, hidden state of previous time step.

        Returns:
            Hidden states of current time step.
        """
        h = x @ self.W + h_prev @ self.U + self.b
        # The return value is actually out
        return torch.tanh(h, out=out)

    def __call__(self, x: Tensor, h_prev: Tensor, out: Tensor):
        return self.forward(x, h_prev, out)
