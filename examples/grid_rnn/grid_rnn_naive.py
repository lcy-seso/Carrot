"""Forward computation of the Grid Long Short Term Memory network for NMT.

Please refer to the paper 'Kalchbrenner, Nal, Ivo Danihelka, and Alex Graves.
Grid long short-term memory. arXiv preprint arXiv:1507.01526 (2015).' for details.
"""
import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from time import time
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from data_utils import build_args_parser, gen_contiguous_input_data
from utils import VanillaRNNCell


class GridRNNNaive(Module):
    def __init__(self, input_size: int, hidden_size: int, grid_dim: int,
                 cells_x: List[Module], cells_y: List[Module]):
        super(GridRNNNaive, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = len(cells_x)

        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

        self.cells_x = cells_x
        self.cells_y = cells_y

    def forward(
            self,
            src_array_batch: Tensor,
            trg_array_batch: Tensor,
            batch_size: int,  # [In]: symbolic constants
            src_lens: List[int],  # [In]: symbolic constants
            trg_lens: List[int]):  # [In]: symbolic constants

        # data parallelism: iterate over samples in a batch.
        sample_id_loop_output: List[List[List[List[Tensor]]]] = []
        for sample_id in range(0, batch_size, 1):
            x = src_array_batch[sample_id]
            y = trg_array_batch[sample_id]

            # dim 1: stack Grid LSTM Cell to form depth.
            d_loop_output: List[List[List[Tensor]]] = []
            for d in range(0, self.depth, 1):
                cell_x = self.cells_x[d]
                cell_y = self.cells_y[d]

                # dim 2: iterate over source sequence length.
                i_loop_output: List[List[Tensor]] = []  # write buffer
                for i in range(0, src_lens[sample_id], 1):
                    # dim 3: iterate over target sequence length.

                    j_loop_output: List[Tensor] = []  # write buffer for j loop
                    for j in range(0, trg_lens[sample_id], 1):
                        if d == 0:
                            x_t = torch.narrow(x, dim=0, start=i, length=1)
                            y_t = torch.narrow(y, dim=0, start=j, length=1)
                        else:
                            x_t = d_loop_output[d - 1][i][(j - 1) * 2]
                            y_t = d_loop_output[d - 1][i][(j - 1) * 2 + 1]

                        if i == 0:
                            state_x = self.init_state
                        else:
                            state_x = i_loop_output[i - 1][(j - 1) * 2]

                        if j == 0:
                            state_y = self.init_state
                        else:
                            state_y = j_loop_output[(j - 1) * 2 + 1]

                        state = torch.cat([state_x, state_y], dim=1)
                        h_x = cell_x(x_t, state_x)
                        h_y = cell_y(y_t, state_y)

                        j_loop_output.append(h_x)
                        j_loop_output.append(h_y)

                    i_loop_output.append(j_loop_output)
                d_loop_output.append(i_loop_output)
                """ multi-directional
                for i in range(src_lens[sample_id], 0, -1):
                    for j in range(trg_lens[sample_id], 0, -1):
                        # ------------- DO something --------------
                """
            sample_id_loop_output.append(d_loop_output)
        return sample_id_loop_output


def test_naive(args, grid_dim, device):
    src_array_batch, trg_array_batch = gen_contiguous_input_data(
        args.batch_size,
        args.input_size,
        args.min_len,
        args.max_len,
        device=device)
    src_lens = [x.size()[0] for x in src_array_batch]
    trg_lens = [x.size()[0] for x in trg_array_batch]

    cells_x = [
        VanillaRNNCell(args.input_size, args.hidden_size).to(device)
        for _ in range(args.depth)
    ]
    cells_y = [
        VanillaRNNCell(args.input_size, args.hidden_size).to(device)
        for _ in range(args.depth)
    ]

    m = GridRNNNaive(args.input_size, args.hidden_size, grid_dim, cells_x,
                     cells_y).to(device)
    m(src_array_batch=src_array_batch,
      trg_array_batch=trg_array_batch,
      batch_size=len(src_array_batch),
      src_lens=src_lens,
      trg_lens=trg_lens)

    print("Finish warmup. Start timing.")
    start = time()
    m(src_array_batch=src_array_batch,
      trg_array_batch=trg_array_batch,
      batch_size=len(src_array_batch),
      src_lens=src_lens,
      trg_lens=trg_lens)
    print("Time elapse = %.6f" % (time() - start))


if __name__ == "__main__":
    args = build_args_parser()
    grid_dim = 2  # Current implementation fixes the grid dim to be 2.

    for device in [
            'cpu',
            'cuda',
    ]:
        print('\n---------------------------------------------------------')
        print(f'Run test on {device}.')
        test_naive(args, grid_dim, device)
