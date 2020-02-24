"""Forward computation of the clockwork RNN.

Please refer to the paper 'Koutnik, Jan, et al. A clockwork rnn.
arXiv preprint arXiv:1402.3511 (2014).' for details.
"""
import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import click
from time import time
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from utils import ClockworkCell, gen_sequence_batch


class StackedClockworkRNN(Module):
    """The Clockwork recurrent network"""

    def __init__(self, cells: List[Module]):
        super(StackedClockworkRNN, self).__init__()
        self.depth = len(cells)
        self.cells = cells

    def forward(self, seq_batch: List[Tensor], seq_lens: List[int]):
        batch_size = len(seq_batch)

        bs_loop_output = []
        for bs in range(batch_size):
            t_loop_output = []
            for t in range(1, seq_lens[bs] + 1, 1):
                d_loop_output = []
                for d, cell in enumerate(self.cells):
                    if d == 0:
                        x_t = torch.narrow(
                            seq_batch[bs], start=t - 1, length=1, dim=0)
                    else:
                        x_t = d_loop_output[d - 1]

                    if t == 1:
                        state = None
                    else:
                        state = t_loop_output[t - 2][d]

                    h_t = cell(t, x_t, state)
                    d_loop_output.append(h_t)
                t_loop_output.append(d_loop_output)
            bs_loop_output.append(t_loop_output)
        return bs_loop_output


@click.command("Build the clockwork RNN model.")
@click.option(
    '--batch_size',
    type=int,
    default=3,
    help=('The batch size defines the number of '
          'samples that will be propagated through the network.'))
@click.option(
    '--input_size',
    type=int,
    default=512,
    help="The number of neurons in RNN's input layer.")
@click.option(
    '--depth', type=int, default=3, help='The number of stacked RNN layer.')
@click.option(
    '--min_len', type=int, default=80, help='The minimum sequence length.')
@click.option(
    '--max_len', type=int, default=100, help='The maximum sequence length.')
@click.option('--block_size', help='The block size.', type=int, default=64)
@click.option(
    '--clock_periods',
    help=('The clock period for CW-RNN cell. Must be delimited by \'.'
          'hidden_size is equal to len(clock_periods) * block_size.'),
    type=str,
    default='1, 2, 4, 8, 16, 32, 64, 128')
def run(input_size, depth, batch_size, min_len, max_len, block_size,
        clock_periods):
    clock_periods = [int(i) for i in clock_periods.split(',')]
    if (len(clock_periods) * block_size != input_size):
        raise ValueError(
            'Current implementation directly stack multiple '
            'clockwork cells which requires '
            'len(clock_periods) * block_size = input_size: '
            f'{len(clock_periods)} * {block_size} != {input_size}.')

    for device in [
            'cpu',
            'cuda',
    ]:
        seq_batch, seq_lens = gen_sequence_batch(batch_size, input_size,
                                                 min_len, max_len, device)
        cells = [
            ClockworkCell(
                input_size=input_size,
                block_size=block_size,
                clock_periods=clock_periods).to(device) for _ in range(depth)
        ]

        m = StackedClockworkRNN(cells)
        m(seq_batch, seq_lens)
        start = time()
        m(seq_batch, seq_lens)
        print(
            f'{device} execution, time elaspe = %.6f (s).' % (time() - start))


if __name__ == "__main__":
    run()
