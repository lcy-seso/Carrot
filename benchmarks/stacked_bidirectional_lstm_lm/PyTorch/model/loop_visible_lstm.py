from abc import ABC, abstractmethod
from math import sqrt

import torch
from torch import empty, sigmoid, tanh, Tensor
from torch.nn import Linear, Module, ModuleList, Parameter
from torch.nn.init import uniform_
from typing import Tuple, List


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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 bidirectional: bool):
        super(LoopVisibleLSTM, self).__init__()

        self.forward_init_input = Linear(input_size, hidden_size)
        self.backward_init_input = Linear(input_size, hidden_size)

        self.forward_cells = ModuleList(
            [self.get_cell(hidden_size, hidden_size) for _ in
             range(num_layers)])

        if bidirectional:
            self.backward_cells = ModuleList(
                [self.get_cell(hidden_size, hidden_size) for _
                 in range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.k = 1 / hidden_size

        self.h_forward = Parameter(torch.Tensor())
        self.c_forward = Parameter(torch.Tensor())
        self.h_backward = Parameter(torch.Tensor())
        self.c_backward = Parameter(torch.Tensor())

    @abstractmethod
    def get_cell(self, input_size: int, hidden_size: int):
        pass

    def forward(self, input: Tensor) -> Tuple[Tensor, None]:
        seq_len = input.size(0)
        batch_size = input.size(1)

        h_forwards = [self.h_forward.new_empty(batch_size, self.hidden_size)
                      for _ in range(self.num_layers)]
        c_forwards = [self.c_forward.new_empty(batch_size, self.hidden_size)
                      for _ in range(self.num_layers)]
        h_backwards = [self.h_backward.new_empty(batch_size, self.hidden_size)
                       for _ in range(self.num_layers)]
        c_backwards = [self.c_backward.new_empty(batch_size, self.hidden_size)
                       for _ in range(self.num_layers)]

        for parameters in [h_forwards, c_forwards, h_backwards, c_backwards]:
            for parameter in parameters:
                uniform_(parameter, -sqrt(self.k), sqrt(self.k))

        # shape of `forward_output` is (seq_len, batch_size, hidden_size)
        forward_output = []

        # shape of `x_t` is (batch_size, input_size)
        for x_t in input:
            # shape of `x` is (batch_size, hidden_size)
            x = self.forward_init_input(x_t)

            # Why there is range-for instead of 'enumerate'? Refer to
            # https://microsoftapc.sharepoint.com/:o:/t/daily_discussion
            # /EvWCa97IgFZNhArch79OV-0Bd5HcPABPKJ8OTNNyjKz3Zg?e=HR4CET

            i = 0
            for forward_cell in self.forward_cells:
                # shape of `h`/`c` is (batch_size, hidden_size)
                h_forwards[i], c_forwards[i] = forward_cell(x, (
                    h_forwards[i], c_forwards[i]))
                x = h_forwards[i]
                i += 1

            forward_output.append(h_forwards[-1])

        if self.bidirectional:

            # shape of `backward_output` is (seq_len, batch_size, hidden_size)
            backward_output: List[Tensor] = []

            for x_t in reversed(input):
                # shape of `x` is (batch_size, hidden_size)
                x = self.backward_init_input(x_t)

                i = 0
                for backward_cell in self.backward_cells:
                    # shape of `h`/`c` is (batch_size, hidden_size)
                    h_backwards[i], c_backwards[i] = backward_cell(x, (
                        h_backwards[i], c_backwards[i]))
                    x = h_backwards[i]
                    i += 1
                backward_output.append(h_backwards[-1])

            output_list: List[Tensor] = []
            for i in range(seq_len):
                output_list.append(torch.cat(
                    (forward_output[i], backward_output[seq_len - 1 - i]), 1))

            output = torch.stack(output_list, dim=0)
        else:
            output = torch.stack(forward_output, dim=0)

        # shape of `output` is (seq_len, batch_size, num_directions*hidden_size)
        return output, None


class DefaultCellLSTM(LoopVisibleLSTM):
    def get_cell(self, input_size: int, hidden_size: int):
        return torch.nn.LSTMCell(input_size, hidden_size)


class JITDefaultCellLSTM(LoopVisibleLSTM):
    def get_cell(self, input_size: int, hidden_size: int):
        return torch.jit.script(torch.nn.LSTMCell(input_size, hidden_size))


class FineGrainedCellLSTM(LoopVisibleLSTM):
    def get_cell(self, input_size: int, hidden_size: int):
        return LSTMCell(input_size, hidden_size)


class JITFineGrainedCellLSTM(LoopVisibleLSTM):
    def get_cell(self, input_size: int, hidden_size: int):
        return torch.jit.script(LSTMCell(input_size, hidden_size))
