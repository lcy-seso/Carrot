from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_normal_ as xavier
from torch.nn import Module

__all__ = [
    "ClockworkCell",
]


class ClockworkCell(Module):
    """Cell function of the Clockwork RNN."""

    def __init__(self,
                 input_size: int,
                 block_size: int,
                 clock_periods: List[int],
                 bias: bool = True):
        """
        Args:
            input_size: int, The number of features in the input tensor.
            hidden_size: int, The number of features in the hidden state.
            clock_periods: List[int], Neurons in clockwork RNN's hidden layer
                          are partitioned into `len(clock_periods)` blocks, and
                          each block is assigned a time period.
            bias: bool, If False, then the cell does not use bias. Default: True
        """
        super(ClockworkCell, self).__init__()

        self.clock_periods = clock_periods
        self.block_num = len(clock_periods)
        self.block_size = block_size

        self.input_size = input_size
        self.hidden_size = self.block_num * block_size

        # ==============================================================
        #    Create the learnable input-to-hidden projection matrix.
        # ==============================================================
        self.input2hidden = []
        for i in range(self.block_num):
            block_row = xavier(Parameter(torch.empty(input_size, block_size)))
            self.register_parameter('i2h%d' % (i), block_row)
            self.input2hidden.append(block_row)

        # ==============================================================
        #    Create the learnable hidden-to-hidden projection matrix.
        # ==============================================================

        # the lower-triangular blocks of the hidden-2-hidden projection are
        # constant tensors with the value zeros.
        self.register_buffer('zero_block', torch.zeros((block_size,
                                                        block_size)))
        self.register_buffer('init_state', torch.zeros(1, self.hidden_size))

        self.hidden2hidden = []
        for i in range(self.block_num):
            for j in range(self.block_num):
                if j >= i:
                    block = xavier(
                        Parameter(torch.empty(block_size, block_size)))
                    self.register_parameter('h2h_%d_%d' % (i, j), block)
                    self.hidden2hidden.append(block)
                else:
                    self.hidden2hidden.append(None)

        if bias:
            self.bias = Parameter(torch.ones(1, self.hidden_size))
        else:
            self.register_parameter('bias', None)

    def forward(self,
                time_step: int,
                input: Tensor,
                state: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input: Tensor, input to current time step.
            state: Tensor, state of last time step.
            time_step: int, current time stpe. NOTE that the counter for
                       time_step MUST start with one.
        """
        if state is None:
            state = self.init_state

        # A hotfix to make constant lower-triangular blocks be on the device.
        for i in range(len(self.hidden2hidden)):
            if self.hidden2hidden[i] is None:
                self.hidden2hidden[i] = self.zero_block

        active_pivot = 0
        for i in range(self.block_num, 0, -1):
            if time_step % self.clock_periods[i - 1] == 0:
                active_pivot = i
                break
        i2h_act = torch.cat(self.input2hidden[0:active_pivot], dim=1)

        end_pos = active_pivot * self.block_size
        prev_unact = torch.narrow(
            state, start=end_pos, length=self.hidden_size - end_pos, dim=1)
        h2h_act = torch.cat(
            self.hidden2hidden[0:active_pivot * self.block_num],
            dim=0).reshape((-1, self.hidden_size))

        hidden_partial = input @ i2h_act + state @ h2h_act.t()
        return torch.tanh(
            torch.cat([hidden_partial, prev_unact], dim=1) + self.bias)
