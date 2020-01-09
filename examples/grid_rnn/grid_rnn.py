import pdb

import numpy as np
from time import time

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import *


def model_def(input_dim, hidden_dim, grid_dim, depth):
    return [
        [
            torch.nn.LSTMCell(input_dim, hidden_dim),
            torch.nn.LSTMCell(input_dim, hidden_dim),
            torch.nn.Linear(hidden_dim * grid_dim,
                            hidden_dim)  # hidden projection
        ] for d in range(depth)
    ]  # the first depth is a 2-D block


def zero_states(hidden_dim):
    return [torch.zeros(1, hidden_dim), torch.zeros(1, hidden_dim)]


def naive_grid_lstm(inputs, depth, batch_size, hidden_dim, cells):
    # data parallelism: iterate over samples in a batch.
    for sample_id in range(batch_size):
        x = inputs[0][sample_id]
        y = inputs[1][sample_id]

        input_dim = x.size()[1]
        src_length = x.size()[0]
        trg_length = y.size()[0]

        # dim 1: stack Grid LSTM Cell to form depth.
        for d in range(depth):
            cell_x = cells[d][0]
            cell_y = cells[d][1]
            hidden_proj = cells[d][2]

            # ===================================== #
            #    Initialize output buffer           #
            # ===================================== #
            outputs = []
            for i in range(src_length + 1):
                outputs.append([[] for j in range(trg_length + 1)])

            for i in range(src_length + 1):
                for j in range(trg_length + 1):
                    if i and j:
                        state_x = []
                        state_y = []
                    else:
                        state_x = zero_states(hidden_dim)
                        state_y = zero_states(hidden_dim)

                    outputs[i][j].append(state_x)
                    outputs[i][j].append(state_y)

            # dim 2: iterate over source sequence length.
            for i in range(1, src_length + 1):
                # ===================================== #
                #    READ access to input array
                # ===================================== #
                x_t = x[i - 1, :].view(1, input_dim)  # this is a vector

                # dim 3: iterate over target sequence length.
                for j in range(1, trg_length + 1):
                    # ===================================== #
                    #    READ access to input array
                    # ===================================== #
                    y_t = y[j - 1, :].view(1, input_dim)  # this is a vector

                    # ===================================== #
                    #    WRITE access to output array
                    # ===================================== #
                    states_x = outputs[i][j - 1][0]
                    states_y = outputs[i - 1][j][1]

                    # ===================================== #
                    #    Cell computation
                    # ===================================== #
                    h_x_prev, c_x_prev = states_x
                    h_y_prev, c_y_prev = states_y
                    h = hidden_proj(torch.cat((h_x_prev, h_y_prev), dim=1))

                    h_x, c_x = cell_x(x_t, (h, c_x_prev))
                    h_y, c_y = cell_y(y_t, (h, c_y_prev))

                    # ===================================== #
                    #    WRITE access to output array
                    # ===================================== #
                    # save hidden and cell state for direction x
                    outputs[i][j][0].append(h_x)
                    outputs[i][j][0].append(c_x)

                    # save hidden and cell state for direction y
                    outputs[i][j][1].append(h_y)
                    outputs[i][j][1].append(c_y)


if __name__ == "__main__":
    args = build_args_parser()

    device = "cpu"
    src_array_batch, trg_array_batch = gen_input_data(
        args.batch_size, args.input_dim, device=device)

    # LSTM cells that are applied to
    cells = model_def(args.input_dim, args.hidden_dim, args.grid_dim,
                      args.depth)
    naive_grid_lstm([src_array_batch, trg_array_batch], args.depth,
                    args.batch_size, args.hidden_dim, cells)
