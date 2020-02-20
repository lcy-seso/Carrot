import pdb

from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_ as xavier
from torch import Tensor
from torch.nn import Parameter

__all__ = [
    "LSTMCell",
    "VanilaRNNCell",
]


class LSTMCell(nn.Module):
    def __init__(self, i2h_shape: List[int], h2h_shape: List[int]):
        super(LSTMCell, self).__init__()
        # learnable paramters for input gate.
        self.Wi = xavier(Parameter(torch.empty(i2h_shape)))
        self.Ui = xavier(Parameter(torch.empty(h2h_shape)))
        self.bi = Parameter(torch.ones(h2h_shape[1]))

        # learnable paramters for forget gate.
        self.Wf = xavier(Parameter(torch.empty(i2h_shape)))
        self.Uf = xavier(Parameter(torch.empty(h2h_shape)))
        self.bf = Parameter(torch.ones(h2h_shape[1]))

        # learnable paramters for cell candidate.
        self.Wg = xavier(Parameter(torch.empty(i2h_shape)))
        self.Ug = xavier(Parameter(torch.empty(h2h_shape)))
        self.bg = Parameter(torch.ones(h2h_shape[1]))

        # learnable paramters for output gate.
        self.Wo = xavier(Parameter(torch.empty(i2h_shape)))
        self.Uo = xavier(Parameter(torch.empty(h2h_shape)))
        self.bo = Parameter(torch.ones(h2h_shape[1]))

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


class VanilaRNNCell(nn.Module):
    """Cell computation of a Vanila RNN.

    This implementation can be automatically differentiated.
    """

    def __init__(self, input_size, hidden_size):
        super(VanilaRNNCell, self).__init__()
        # learnable paramters
        self.W = xavier(Parameter(torch.empty(input_size, hidden_size)))
        self.U = xavier(Parameter(torch.empty(hidden_size, hidden_size)))
        self.b = Parameter(torch.ones(hidden_size))

        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

    def forward(self, input: Tensor, h_prev: Tensor) -> Tensor:
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


class VanilaRNNCell_(object):
    """Inplace version. Cell computation of a Vanila RNN.

    NOTE: This implementation cannot be automatically differentiated.
    """

    def __init__(self, input_size, hidden_size):
        super(VanilaRNNCell_, self).__init__()
        # learnable paramters
        self.W = xavier(torch.empty(input_size, hidden_size))
        self.U = xavier(torch.empty(hidden_size, hidden_size))
        self.b = torch.ones(1, hidden_size)

        self.init_state = torch.zeros((1, hidden_size))

    def to(self, device):
        self.W = self.W.to(device)
        self.U = self.U.to(device)
        self.b = self.b.to(device)
        self.init_state = self.init_state.to(device)

    def forward(self, x: Tensor, h_prev: Tensor, out: Tensor):
        """Cell computation of a Vanila RNN.
        Args:
            input, Tensor, input to current time step with a shape of
                [batch_size, input_size].
            h_prev, Tuple, hidden state of previous time step.

        Returns:
            Hidden states of current time step.
        """
        h_prev = self.init_state if h_prev is None else h_prev
        torch.addmm(beta=0, input=out, alpha=1, mat1=x, mat2=self.W, out=out)
        h = torch.mm(h_prev, self.U)
        torch.add(h, other=0., out=out)
        return torch.add(self.b, other=0., out=out)

    def __call__(self, x: Tensor, h_prev: Tensor, out: Tensor):
        return self.forward(x, h_prev, out)


def test_VanilaRNNCell(input_size=7, hidden_size=11):
    batch_size = 3
    for device in [
            "cpu",
            "cuda",
    ]:
        out = torch.empty(batch_size, hidden_size, device=device)
        x = torch.randn(batch_size, input_size, device=device)

        cell = VanilaRNNCell(input_size, hidden_size).to(device)

        cell_ = VanilaRNNCell_(input_size, hidden_size)
        cell_.to(device)

        h = cell(x, None)
        h_ = cell_(x, None, out)

        h = cell(x, h)
        h_ = cell_(x, h_, out)


if __name__ == "__main__":
    test_VanilaRNNCell()
