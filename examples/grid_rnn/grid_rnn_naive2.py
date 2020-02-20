#!/usr/bin/env python3
#coding=utf-8
"""Forward computation of the Grid Long Short Term Memory network for NMT.

Please refer to the paper 'Kalchbrenner, Nal, Ivo Danihelka, and Alex Graves.
Grid long short-term memory. arXiv preprint arXiv:1507.01526 (2015).' for details.
"""
import pdb

from time import time
from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from rnncell import *
from utils import *


def rnn_net(
        src_array_batch: List[Tensor],
        trg_array_batch: List[Tensor],
        cells_x: List[Module],  # [In]: readonly, sub computation graphs
        cells_y: List[Module],  # [In]: readonly, sub computation graphs
        batch_size: int,  # [In]: symbolic constants
        depth: int,  # [In]: symbolic constants
        src_lens: List[int],  # [In]: symbolic constants
        trg_lens: List[int],  # [In]: symbolic constants
        input_dim: int,  # [In]: symbolic constants
        hidden_dim: int,
        grid_dim: int,
        device: str):
    # data parallelism: iterate over samples in a batch.
    sample_id_loop_output = []
    for sample_id in range(0, batch_size, 1):
        x = src_array_batch[sample_id]
        y = trg_array_batch[sample_id]

        # dim 1: stack Grid LSTM Cell to form depth.
        d_loop_output = []
        for d in range(0, depth, 1):
            cell_x = cells_x[d]
            cell_y = cells_y[d]

            # dim 2: iterate over source sequence length.
            i_loop_output = []  # write buffer for i loop
            for i in range(0, src_lens[sample_id], 1):
                # dim 3: iterate over target sequence length.

                j_loop_output = []  # write buffer for j loop
                for j in range(0, trg_lens[sample_id], 1):
                    if d == 0:
                        x_t = torch.narrow(x, dim=0, start=i, length=1)
                        y_t = torch.narrow(y, dim=0, start=j, length=1)
                    else:
                        x_t = d_loop_output[d - 1][i][j]
                        y_t = d_loop_output[d - 1][i][j]

                    if i == 0:
                        state_x = torch.zeros(1, hidden_dim, device=device)
                    else:
                        state_x = i_loop_output[i - 1][(j - 1) * 2]

                    if j == 0:
                        state_y = torch.zeros(1, hidden_dim, device=device)
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


def naive_grid_lstm(
        src_array_batch: List[Tensor],
        trg_array_batch: List[Tensor],
        cells_x: List[Module],
        cells_y: List[Module],
        input_dim: int,
        hidden_dim: int,
        device: str,
):
    """
    Args:
        src_array_batch: List[Tensor], input array for read access only,
                                       the source sequence batch.
        trg_array_batch: List[Tensor], input array for read access only,
                                       the target sequence batch.
        cells: List[callable], input array for read access only,
                                       the cells to form depth.
        input_dim: int, the input dimension of RNN cells.
        hidden_dim: int, the output dimension of RNN cells.
    """
    # batch_size and depth are constants independent of data.
    batch_size = len(src_array_batch)
    depth = len(cells_x)

    # ==================================================================== #
    #                 Initialize output buffer                             #
    # ==================================================================== #
    src_lens = [x.size()[0] for x in src_array_batch]
    trg_lens = [x.size()[0] for x in trg_array_batch]

    rnn_net(
        src_array_batch=src_array_batch,
        trg_array_batch=trg_array_batch,  # [In]
        cells_x=cells_x,
        cells_y=cells_y,  # [In]
        batch_size=batch_size,  # loop1
        depth=depth,  # loop2
        src_lens=src_lens,  # loop3
        trg_lens=trg_lens,  # loop4
        grid_dim=2,  # loop5
        device=device,
        input_dim=input_dim,
        hidden_dim=hidden_dim)


if __name__ == "__main__":
    args = build_args_parser()
    grid_dim = 2  # Current implementation fixes the grid dim to be 2.

    for device in [
            # 'cpu',
            'cuda',
    ]:

        print('\n---------------------------------------------------------')
        print(f'Run test on {device}.')

        src_array_batch, trg_array_batch = gen_contiguous_input_data(
            args.batch_size,
            args.input_dim,
            args.min_len,
            args.max_len,
            device=device)
        cells_x = [
            VanilaRNNCell(args.input_dim, args.hidden_dim).to(device)
            for _ in range(args.depth)
        ]
        cells_y = [
            VanilaRNNCell(args.input_dim, args.hidden_dim).to(device)
            for _ in range(args.depth)
        ]
        naive_grid_lstm(
            src_array_batch=src_array_batch,
            trg_array_batch=trg_array_batch,
            cells_x=cells_x,
            cells_y=cells_y,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            device=device)
        print("Finish warmup. Start timing.")

        start = time()
        naive_grid_lstm(
            src_array_batch=src_array_batch,
            trg_array_batch=trg_array_batch,
            cells_x=cells_x,
            cells_y=cells_y,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            device=device)
        print("Time elapse = %.6f" % (time() - start))
