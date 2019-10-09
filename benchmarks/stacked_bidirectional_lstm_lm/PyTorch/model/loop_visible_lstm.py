from abc import ABC, abstractmethod
from math import sqrt

import torch
from torch import empty, sigmoid, tanh, Tensor
from torch.nn import Module, Parameter
from torch.nn.init import uniform_
from typing import Tuple


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.W_ii = Parameter(empty(input_size, hidden_size))
        self.b_ii = Parameter(empty(hidden_size))

        self.W_if = Parameter(empty(input_size, hidden_size))
        self.b_if = Parameter(empty(hidden_size))

        self.W_ig = Parameter(empty(input_size, hidden_size))
        self.b_ig = Parameter(empty(hidden_size))

        self.W_io = Parameter(empty(input_size, hidden_size))
        self.b_io = Parameter(empty(hidden_size))

        self.W_hi = Parameter(empty(hidden_size, hidden_size))
        self.b_hi = Parameter(empty(hidden_size))

        self.W_hf = Parameter(empty(hidden_size, hidden_size))
        self.b_hf = Parameter(empty(hidden_size))

        self.W_hg = Parameter(empty(hidden_size, hidden_size))
        self.b_hg = Parameter(empty(hidden_size))

        self.W_ho = Parameter(empty(hidden_size, hidden_size))
        self.b_ho = Parameter(empty(hidden_size))

        k = 1 / hidden_size
        for parameter in self.parameters():
            uniform_(parameter, -sqrt(k), sqrt(k))

    def forward(self, input: Tensor, hx: Tuple[Tensor, Tensor]) \
            -> Tuple[Tensor, Tensor]:
        # shape of `h_0`/`c_0` is (batch_size, hidden_size)
        h_0, c_0 = hx

        # shape of `i` is (batch_size, hidden_size)
        i: Tensor = sigmoid(
            input.mm(self.W_ii) + self.b_ii + h_0.mm(self.W_hi) + self.b_hi)

        # shape of `f` is (batch_size, hidden_size)
        f: Tensor = sigmoid(
            input.mm(self.W_if) + self.b_if + h_0.mm(self.W_hf) + self.b_hf)

        # shape of `g` is (batch_size, hidden_size)
        g: Tensor = tanh(
            input.mm(self.W_ig) + self.b_ig + h_0.mm(self.W_hg) + self.b_hg)

        # shape of `o` is (batch_size, hidden_size)
        o: Tensor = sigmoid(
            input.mm(self.W_io) + self.b_io + h_0.mm(self.W_ho) + self.b_ho)

        # shape of `h_1`/`c_1` is (batch_size, hidden_size)
        c_1 = f.mul(c_0) + i.mul(g)
        h_1 = o.mul(c_1)

        return h_1, c_1


class LoopVisibleLSTM(ABC, Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool):
        super(LoopVisibleLSTM, self).__init__()
        self.cell = self.get_cell(input_size, hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.k = 1 / hidden_size

        self.register_buffer('h', torch.Tensor())
        self.register_buffer('c', torch.Tensor())

    @abstractmethod
    def get_cell(self, input_size: int, hidden_size: int):
        pass

    def forward(self, input: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_size = input.size(1)

        self.h = self.h.new_empty(batch_size, self.hidden_size)
        self.c = self.c.new_empty(batch_size, self.hidden_size)
        uniform_(self.h, -sqrt(self.k), sqrt(self.k))
        uniform_(self.c, -sqrt(self.k), sqrt(self.k))

        # shape of `forward_output` is (seq_len, batch_size, hidden_size)
        forward_output = []

        # shape of `x_t` is (batch_size, input_size)
        for x_t in input:
            # shape of `h`/`c` is (batch_size, hidden_size)
            self.h, self.c = self.cell(x_t, (self.h, self.c))
            forward_output.append(self.h)

        if self.bidirectional:
            # shape of `backward_output` is (seq_len, batch_size, hidden_size)
            backward_output = []
            for x_t in reversed(input):
                # shape of `h`/`c` is (batch_size, hidden_size)
                self.h, self.c = self.cell(x_t, (self.h, self.c))
                backward_output.append(self.h)

            output = torch.stack(list(map(lambda x: torch.cat((x[0], x[1]), 1),
                                          zip(forward_output,
                                              reversed(backward_output)))),
                                 dim=0)
        else:
            output = torch.stack(forward_output, dim=0)

        # shape of `output` is (seq_len, batch_size, num_directions*hidden_size)
        return output, (self.h, self.c)


class DefaultCellLSTM(LoopVisibleLSTM):
    def get_cell(self, input_size: int, hidden_size: int):
        return torch.nn.LSTMCell(input_size, hidden_size)


class FineGrainedCellLSTM(LoopVisibleLSTM):
    def get_cell(self, input_size: int, hidden_size: int):
        return LSTMCell(input_size, hidden_size)
