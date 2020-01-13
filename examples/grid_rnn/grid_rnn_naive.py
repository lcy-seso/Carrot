from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from utils import *


def naive_grid_lstm(src_array_batch: List[Tensor],
                    trg_array_batch: List[Tensor], cells: List[Module],
                    input_dim: int, hidden_dim: int, device: str):
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
    depth = len(cells)

    # data parallelism: iterate over samples in a batch.
    for sample_id in range(0, batch_size, 1):
        x = src_array_batch[sample_id]
        y = trg_array_batch[sample_id]

        src_length = x.size()[0]
        trg_length = y.size()[0]

        # ===================================== #
        #    Initialize output buffer           #
        # ===================================== #
        outputs = []
        for i in range(depth):
            outputs.append(
                init_out_buff(src_length, trg_length, hidden_dim, device))

        # dim 1: stack Grid LSTM Cell to form depth.
        for d in range(0, depth, 1):
            # dim 2: iterate over source sequence length.
            for i in range(1, src_length + 1, 1):
                # dim 3: iterate over target sequence length.
                for j in range(1, trg_length + 1, 1):
                    # ===================================== #
                    #    READ access to input arrays to     #
                    #    get inputs to iteration (d, i, j)  #
                    # ===================================== #
                    cell_x = cells[d][0]
                    cell_y = cells[d][1]

                    output_d = outputs[d]

                    if d == 0:
                        x_t = x[i - 1, :].view(1, input_dim)
                        y_t = y[j - 1, :].view(1, input_dim)
                    else:
                        # ================================================ #
                        # NOTE:  when d > 0, x_t and y_t are inputs in the #
                        # depth direction, though we do not apply          #
                        # learnable projections to the depth direction in  #
                        # current implementation, the depth direction can  #
                        # also form recurrent computation.                 #
                        # ================================================ #
                        x_t = outputs[d - 1][i][j][0][0]
                        y_t = outputs[d - 1][i][j][1][0]
                    states_x = output_d[i][j - 1][0]
                    states_y = output_d[i - 1][j][1]

                    # =========================================== #
                    #    cell computation in iteration (d, i, j)  #
                    # =========================================== #
                    h_x_prev, c_x_prev = states_x
                    h_y_prev, c_y_prev = states_y

                    h = torch.cat((h_x_prev, h_y_prev), dim=1)
                    h_x, c_x = cell_x(x_t, (h, c_x_prev))
                    h_y, c_y = cell_y(y_t, (h, c_y_prev))

                    # ================================================== #
                    #    WRITE access to the output array to             #
                    #    save produced results in iteration (d, i, j)    #
                    # ================================================== #

                    output_d[i][j][0].append(h_x)  # hidden for direction x
                    output_d[i][j][0].append(c_x)  # cell for direction x

                    output_d[i][j][1].append(h_y)  # hidden for direction y
                    output_d[i][j][1].append(c_y)  # cell for direction y


if __name__ == "__main__":
    args = build_args_parser()

    for device in [
            "cpu",
            "cuda:0",
    ]:
        cells = model_def(args.input_dim, args.hidden_dim, args.grid_dim,
                          args.depth, device)
        src_array_batch, trg_array_batch = gen_input_data(
            args.batch_size, args.input_dim, device=device)

        naive_grid_lstm(src_array_batch, trg_array_batch, cells,
                        args.input_dim, args.hidden_dim, device)
