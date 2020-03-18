"""Naive implementation of the mogrifier LSTM.

Please refere to the paper 'Melis G, Kočis T, Blunsom P. Mogrifier lstm[J].
arXiv preprint arXiv:1909.01792, 2019' for detailed information.
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

from utils import MogLSTMCell, gen_sequence_batch


class StackedRNN(Module):
    """The Clockwork recurrent network"""

    def __init__(self, cells: List[Module]):
        super(StackedRNN, self).__init__()
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
                        x_t = d_loop_output[d - 1][0]

                    if t == 1:
                        states = None
                    else:
                        states = t_loop_output[t - 2][d]

                    _, states = cell(x_t, states)
                    d_loop_output.append(states)
                t_loop_output.append(d_loop_output)
            bs_loop_output.append(t_loop_output)
        return bs_loop_output


@click.command('Build the Mogrifier RNN model.')
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
    '--hidden_size',
    type=int,
    default=512,
    help="The number of neurons in RNN's hidden layer.")
@click.option(
    '--depth', type=int, default=3, help='The number of stacked RNN layers.')
@click.option(
    '--mog_iteration',
    type=int,
    default=5,
    help='The number of rounds for mogrifications.')
@click.option(
    '--min_len', type=int, default=80, help='The minimum sequence length.')
@click.option(
    '--max_len', type=int, default=100, help='The maximum sequence length.')
def run(input_size, hidden_size, depth, batch_size, min_len, max_len,
        mog_iteration):

    for device in [
            'cpu',
            'cuda',
    ]:
        seq_batch, seq_lens = gen_sequence_batch(batch_size, input_size,
                                                 min_len, max_len, device)

        m = StackedRNN(cells=[
            MogLSTMCell(input_size, hidden_size, mog_iteration).to(device)
            for _ in range(depth)
        ])
        m(seq_batch, seq_lens)
        start = time()
        m(seq_batch, seq_lens)
        print((f'{device} execution, '
               'time elaspe = %.6f (s).') % (time() - start))


if __name__ == "__main__":
    run()
