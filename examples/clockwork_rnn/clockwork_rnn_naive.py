"""Forward computation of the clockwork RNN.

Please refer to the paper "Koutnik, Jan, et al. "A clockwork rnn."
arXiv preprint arXiv:1402.3511 (2014)." for details.
"""
import time

from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from clockwork_cell import ClockworkCell
from utils import *


def stacked_clockwork_rnn(seq_batch: List[Tensor], input_dim: int,
                          hidden_dim: int, cells: List[Module], device: str):
    depth = len(cells)
    batch_size = len(seq_batch)

    outputs = [[torch.zeros((1, hidden_dim), device=device)]
               for _ in range(depth)]
    for i in range(0, batch_size, 1):
        x = seq_batch[i]
        seq_len = x.size()[0]
        for time_step in range(1, seq_len + 1, 1):
            x_t = torch.narrow(x, start=time_step - 1, length=1, dim=0)
            for j in range(depth):
                cell = cells[j]
                state = outputs[j][time_step - 1]

                h_t = cell(x_t, state, time_step)
                outputs[j].append(h_t)


if __name__ == "__main__":
    args = build_args_parser()
    clock_periods = [int(i) for i in args.clock_periods.split(',')]
    for device in [
            "cpu",
            "cuda",
    ]:
        seq_batch = gen_input_data(args.batch_size, args.input_size, device)
        cells = [
            ClockworkCell(
                input_size=args.input_size,
                block_size=args.block_size,
                clock_periods=clock_periods).to(device)
            for _ in range(args.depth)
        ]

        stacked_clockwork_rnn(seq_batch, args.input_size,
                              len(clock_periods) * args.block_size, cells,
                              device)
