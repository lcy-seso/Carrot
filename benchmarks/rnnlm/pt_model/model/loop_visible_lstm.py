from abc import ABC, abstractmethod
from math import sqrt

import torch
from torch import empty, sigmoid, tanh, Tensor
from torch.nn import Linear, Module, Parameter
from torch.nn.init import uniform_
from typing import Tuple, List

index_DefaultCellLSTM = 0
index_JITDefaultCellLSTM = 1
index_FineGrainedCellLSTM = 2
index_JITFineGrainedCellLSTM = 3


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
        h_1 = o.mul(tanh(c_1))

        return h_1, c_1


class InnerLoop(Module):
    def __init__(self, hidden_size: int, lstm_index: int):
        super(InnerLoop, self).__init__()
        if lstm_index == index_DefaultCellLSTM:
            self.cell = torch.nn.LSTMCell(hidden_size, hidden_size)
        elif lstm_index == index_FineGrainedCellLSTM:
            self.cell = LSTMCell(hidden_size, hidden_size)
        elif lstm_index == index_JITDefaultCellLSTM:
            self.cell = torch.jit.script(
                torch.nn.LSTMCell(hidden_size, hidden_size))
        elif lstm_index == index_JITFineGrainedCellLSTM:
            self.cell = torch.jit.script(LSTMCell(hidden_size, hidden_size))

    def forward(self, input: List[Tensor], h, c) -> List[Tensor]:
        result = []

        for x in input:
            # shape of `x` is (batch_size, hidden_size)
            h, c = self.cell(x, (h, c))
            result.append(h)

        return result


class OuterLoop(Module):
    def __init__(self, num_layers: int, hidden_size: int, lstm_index: int):
        super(OuterLoop, self).__init__()

        # We can't use `ModuleList` here
        # for [[jit] Can't index nn.ModuleList in script function]
        # (https://github.com/pytorch/pytorch/issues/16123).
        # Therefore, we use hard code there for num_layers==3.
        self.inner_loop_1 = InnerLoop(hidden_size, lstm_index)
        self.inner_loop_2 = InnerLoop(hidden_size, lstm_index)
        self.inner_loop_3 = InnerLoop(hidden_size, lstm_index)

        if num_layers != 3:
            raise RuntimeError('`num_layers` should be 3')

        self.num_layers = num_layers

    def forward(self, input: List[Tensor], h_forwards: List[Tensor],
                c_forwards: List[Tensor]) -> List[Tensor]:

        for i in range(self.num_layers):
            if i == 0:
                input = self.inner_loop_1(input, h_forwards[i], c_forwards[i])
            elif i == 1:
                input = self.inner_loop_2(input, h_forwards[i], c_forwards[i])
            elif i == 2:
                input = self.inner_loop_3(input, h_forwards[i], c_forwards[i])

        return input


class LoopVisibleLSTM(ABC, Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 bidirectional: bool):
        super(LoopVisibleLSTM, self).__init__()

        self.forward_init_input = Linear(input_size, hidden_size)
        self.backward_init_input = Linear(input_size, hidden_size)

        if bidirectional:
            raise RuntimeError('`bidirectional` is not supported now')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.k = 1 / hidden_size

        self.h_forward = Parameter(torch.Tensor())
        self.c_forward = Parameter(torch.Tensor())

        self.outer_loop = OuterLoop(num_layers, hidden_size,
                                    self.get_lstm_index())

    @abstractmethod
    def get_lstm_index(self) -> int:
        pass

    def forward(self, input: Tensor) -> Tuple[Tensor, None]:
        batch_size = input.size(1)

        h_forwards = [
            self.h_forward.new_empty(batch_size, self.hidden_size)
            for _ in range(self.num_layers)
        ]
        c_forwards = [
            self.c_forward.new_empty(batch_size, self.hidden_size)
            for _ in range(self.num_layers)
        ]

        for parameters in [h_forwards, c_forwards]:
            for parameter in parameters:
                uniform_(parameter, -sqrt(self.k), sqrt(self.k))

        if self.bidirectional:
            raise RuntimeError('`bidirectional` is not supported now')

        input = self.forward_init_input(input)

        # We can't use `input = [x for x in input]` here
        # for [[JIT] list comprehensions over tensors does not work]
        # (https://github.com/pytorch/pytorch/issues/27255)
        outer_loop_input = []
        for x in input:
            outer_loop_input.append(x)
        forward_output = self.outer_loop(outer_loop_input, h_forwards,
                                         c_forwards)

        output = torch.stack(forward_output, dim=0)

        # shape of `output` is (seq_len, batch_size, num_directions*hidden_size)
        return output, None


class DefaultCellLSTM(LoopVisibleLSTM):
    def get_lstm_index(self) -> int:
        return index_DefaultCellLSTM


class JITDefaultCellLSTM(LoopVisibleLSTM):
    def get_lstm_index(self):
        return index_JITDefaultCellLSTM


class FineGrainedCellLSTM(LoopVisibleLSTM):
    def get_lstm_index(self):
        return index_FineGrainedCellLSTM


class JITFineGrainedCellLSTM(LoopVisibleLSTM):
    def get_lstm_index(self):
        return index_JITFineGrainedCellLSTM
