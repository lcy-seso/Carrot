from itertools import groupby
import time

from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from utils import *


def grid_lstm_skew_to_outermost_loop(
        src_array_batch: List[Tensor], trg_array_batch: List[Tensor],
        cells: List[Module], input_dim: int, hidden_dim: int, device: str):
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
    batch_size = len(src_array_batch)
    depth = len(cells)

    # ==================================================================== #
    #                 Initialize output buffer                             #
    # ==================================================================== #
    src_lens = [src_array_batch[i].size()[0] for i in range(batch_size)]
    trg_lens = [trg_array_batch[i].size()[0] for i in range(batch_size)]

    # `outputs` is the output buffer. A nested array with a depth 4 is used.
    outputs: List[List[List[List[Tensor]]]] = []
    for src_length, trg_length in zip(src_lens, trg_lens):
        outputs.append([])
        for i in range(depth):
            outputs[-1].append(
                init_out_buff(src_length, trg_length, hidden_dim, device))

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
        src_length = src_array_batch[sample_id].size()[0]
        trg_length = trg_array_batch[sample_id].size()[0]
        for d in range(0, depth, 1):
            for i in range(1, src_length + 1, 1):
                for j in range(1, trg_length + 1, 1):
                    trans_points.append([d + i + j, sample_id, d, i])
    trans_points = sorted(trans_points, key=lambda x: x[0], reverse=False)

    gather_time = 0.
    compute_time = 0.
    scatter_time = 0.

    for z, value in groupby(trans_points, key=lambda x: x[0]):
        cell_points = sorted(
            list(value), key=lambda x: x[2], reverse=False)  # sort by depth
        for d, data_points in groupby(cell_points, key=lambda x: x[2]):

            # ===================================================== #
            #            Gather parallelizable inputs               #
            # ===================================================== #
            data_points = list(data_points)
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
                    start_gather = time.time()
                    x_t.append(src_array_batch[sample_id][i - 1, :].view(
                        1, input_dim))
                    y_t.append(trg_array_batch[sample_id][j - 1, :].view(
                        1, input_dim))
                    gather_time += (time.time() - start_gather)
                else:
                    x_t.append(outputs[sample_id][d - 1][i][j][0][0])
                    y_t.append(outputs[sample_id][d - 1][i][j][1][0])

                states_x.append(outputs[sample_id][d][i][j - 1][0])
                states_y.append(outputs[sample_id][d][i - 1][j][1])

            start_gather = time.time()
            # ========================================================== #
            #   Batch parallelizable inputs and perform cell computation #
            # ========================================================== #
            x_t = __batch(x_t)
            y_t = __batch(y_t)

            h_x_prev = __batch([state[0] for state in states_x])
            c_x_prev = __batch([state[1] for state in states_x])
            h_y_prev = __batch([state[0] for state in states_y])
            c_y_prev = __batch([state[1] for state in states_y])

            start_compute = time.time()
            gather_time += (start_compute - start_gather)

            h = torch.cat((h_x_prev, h_y_prev), dim=1)
            h_x, c_x = cells[d][0](x_t, (h, c_x_prev))
            h_y, c_y = cells[d][1](y_t, (h, c_y_prev))

            start_scatter = time.time()
            compute_time += (start_scatter - start_compute)

            # ========================================================== #
            #           Scatter outputs                                  #
            # ========================================================== #
            h_x = __unbatch(h_x)
            c_x = __unbatch(c_x)
            h_y = __unbatch(h_y)
            c_y = __unbatch(c_y)
            scatter_time += (time.time() - start_scatter)

            for num, (sample_id, d, i, j) in enumerate(gather_points):
                outputs[sample_id][d][i][j][0].append(h_x[num])
                outputs[sample_id][d][i][j][0].append(c_x[num])

                outputs[sample_id][d][i][j][1].append(h_y[num])
                outputs[sample_id][d][i][j][1].append(c_y[num])

    return gather_time, compute_time, scatter_time


if __name__ == "__main__":
    run_test(grid_lstm_skew_to_outermost_loop)
