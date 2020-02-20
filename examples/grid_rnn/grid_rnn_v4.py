#!/usr/bin/env python3
#coding=utf-8
import pdb

import random
from itertools import groupby
from time import time

from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from utils import *
from rnncell import VanilaRNNCell


def lens_to_start_pos(lens):
    start_pos = [0]
    for i, l in enumerate(lens):
        start_pos.append(start_pos[i] + lens[i])
    return start_pos


def gen_one_batch(batch_size, input_dim, min_len, max_len, device):
    seq_len = [random.randint(min_len, max_len) for _ in range(batch_size)]
    return torch.randn(sum(seq_len), input_dim, device=device), seq_len


def __batch_input(x, indices, device, dim=0):
    return torch.index_select(
        x, dim, torch.tensor(indices, device=device, requires_grad=False))


def grid_lstm_v4(
        src_array_batch: Tensor,  # [In]: Tensor
        src_lens: List[Tensor],  # [In]: symbolic constants
        trg_array_batch: Tensor,  # [In]: Tensor
        trg_lens: List[Tensor],  # [In]: symbolic constants
        cells_x: List[Module],  # [In]: sub-graphs
        cells_x: List[Module],  # [In]: sub-graphs
        input_dim: int,
        hidden_dim: int,
        outputs: Tensor,  # [Out]
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
    batch_size = len(src_lens)
    depth = len(cells)

    src_start_pos = lens_to_start_pos(src_lens)
    trg_start_pos = lens_to_start_pos(trg_lens)

    # ===================================================================== #
    # Wavefront transformation to the entire loop nesting.
    # 4D transformation matrix for the wavefront transformation is:
    #    [[0, 1, 1, 1]
    #     [1, 0, 0, 0]
    #     [0, 1, 0, 0]
    #     [0, 0, 1, 0]]
    # After applying wavefront transformation, the uttermost `m` loop is
    # sequential, while all the inner loops are able to execute in parallel.
    # This is still NOT the optimal parallelisms.
    # ===================================================================== #

    trans_points = []
    for sample_id in range(0, batch_size, 1):

        src_length = src_lens[sample_id]
        trg_length = trg_lens[sample_id]

        for d in range(0, depth, 1):
            for i in range(1, src_length + 1, 1):
                for j in range(1, trg_length + 1, 1):
                    trans_points.append([d + i + j, sample_id, d, i])
    trans_points = sorted(trans_points, key=lambda x: x[0], reverse=False)

    for z, value in groupby(trans_points, key=lambda x: x[0]):
        cell_points = sorted(
            list(value), key=lambda x: x[2], reverse=False)  # sort by depth

        for d, data_points in groupby(cell_points, key=lambda x: x[2]):
            # ===================================================== #
            #            Gather parallelizable inputs
            # ===================================================== #
            data_points = list(data_points)
            indices_x: List[int] = []
            indices_y: List[int] = []

            x_t: List[Tensor] = []
            y_t: List[Tensor] = []

            states_x: List[List[Tensor], List[Tensor]] = []
            states_y: List[List[Tensor], List[Tensor]] = []
            gather_points = []
            for p in data_points:
                sample_id = p[1]
                i = p[3]
                j = p[0] - d - p[3]
                gather_points.append([sample_id, d, i, j])  # write position

                if d == 0:
                    indices_x.append(src_start_pos[sample_id] + i - 1)
                    indices_y.append(trg_start_pos[sample_id] + j - 1)
                    # x_t.append(src_array_batch[sample_id][i - 1, :].view(
                    #     1, input_dim))
                    # y_t.append(trg_array_batch[sample_id][j - 1, :].view(
                    #     1, input_dim))
                else:
                    x_t.append(outputs[sample_id][d - 1][i][j][0][0])
                    y_t.append(outputs[sample_id][d - 1][i][j][1][0])

                states_x.append(outputs[sample_id][d][i][j - 1][0])
                states_y.append(outputs[sample_id][d][i - 1][j][1])

            if d == 0:
                x_t = __batch_input(src_array_batch, indices_x, device)
                y_t = __batch_input(trg_array_batch, indices_y, device)
            else:
                x_t = __batch(x_t)
                y_t = __batch(y_t)

            h_x_prev = __batch([state[0] for state in states_x])
            c_x_prev = __batch([state[1] for state in states_x])
            h_y_prev = __batch([state[0] for state in states_y])
            c_y_prev = __batch([state[1] for state in states_y])

            # ========================================================== #
            #   Cell computation.
            # ========================================================== #
            h = torch.cat((h_x_prev, h_y_prev), dim=1)
            h_x = cells_x[d](x_t, (h, c_x_prev))
            h_y = cells_y[d](y_t, (h, c_y_prev))

            # ========================================================== #
            #           Scatter outputs
            # ========================================================== #
            h_x = __unbatch(h_x)
            c_x = __unbatch(c_x)
            h_y = __unbatch(h_y)
            c_y = __unbatch(c_y)

            for num, (sample_id, d, i, j) in enumerate(gather_points):
                outputs[sample_id][d][i][j][0].append(h_x[num])
                outputs[sample_id][d][i][j][0].append(c_x[num])

                outputs[sample_id][d][i][j][1].append(h_y[num])
                outputs[sample_id][d][i][j][1].append(c_y[num])


def init_out_buff(src_lens, trg_lens):
    # ==================================================================== #
    #                 Initialize output buffer                             #
    # ==================================================================== #
    # `outputs` is the output buffer. A nested array with a depth 4 is used.
    outputs: List[List[List[List[Tensor]]]] = []
    for src_length, trg_length in zip(src_lens, trg_lens):
        outputs.append([])
        for i in range(depth):
            outputs[-1].append(
                init_out_buff(src_length, trg_length, hidden_dim, device))


if __name__ == "__main__":
    args = build_args_parser()
    grid_dim = 2  # Current implementation fixes the grid dim to be 2.

    # min_len = 5
    # max_len = 20
    # batch_size = 4

    min_len = args.min_len
    max_len = args.max_len
    batch_size = args.batch_size

    for device in [
            # 'cpu',
            'cuda',
    ]:

        print('\n---------------------------------------------------------')
        print(f'Run test on {device}.')

        cells_x = []
        cells_y = []
        for i in range(depth):
            cell_x = VanilaRNNCell(args.input_dim, args.hidden_dim)
            cells_x.append(cell_x.to(device))

            cell_y = VanilaRNNCell(args.input_dim, args.hidden_dim)
            cells_y.append(cell_y.to(device))

        src_array_batch, src_lens = gen_one_batch(
            batch_size,
            args.input_dim,
            device=device,
            min_len=min_len,
            max_len=max_len)
        trg_array_batch, trg_lens = gen_one_batch(
            batch_size,
            args.input_dim,
            device=device,
            min_len=min_len,
            max_len=max_len)

        outputs = torch.empty(sum(trg_lens))

        grid_lstm_v4(
            src_array_batch,
            src_lens,
            trg_array_batch,
            trg_lens,
            cells,
            args.input_dim,
            args.hidden_dim,
            device=device)
        print("Finish warmup. Start timing.")

        start = time()
        grid_lstm_v4(
            src_array_batch,
            src_lens,
            trg_array_batch,
            trg_lens,
            cells_x,
            cells_y,
            args.input_dim,
            args.hidden_dim,
            device=device)
        print("Time elapse = %.6f" % (time() - start))

    # import cProfile
    # import pstats
    # import io

    # Profile python
    # pr = cProfile.Profile()
    # pr.enable()
    # run_test(grid_lstm_skew_to_outermost_loop)
    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
