import torch

from utils import *


def naive_grid_lstm(inputs, depth, batch_size, hidden_dim, cells, device):
    # data parallelism: iterate over samples in a batch.
    for sample_id in range(batch_size):
        x = inputs[0][sample_id]
        y = inputs[1][sample_id]

        input_dim = x.size()[1]
        src_length = x.size()[0]
        trg_length = y.size()[0]

        # dim 1: stack Grid LSTM Cell to form depth.
        outputs = []
        for d in range(depth):
            cell_x = cells[d][0]
            cell_y = cells[d][1]
            hidden_proj = cells[d][2]

            # ===================================== #
            #    Initialize output buffer           #
            # ===================================== #
            outputs.append(
                init_out_buff(src_length, trg_length, hidden_dim, device))
            output_d = outputs[-1]
            # dim 2: iterate over source sequence length.
            for i in range(1, src_length + 1):
                # dim 3: iterate over target sequence length.
                for j in range(1, trg_length + 1):
                    # ===================================== #
                    #    READ access to input/output array
                    # ===================================== #
                    if d == 0:
                        x_t = x[i - 1, :].view(1, input_dim)
                        y_t = y[j - 1, :].view(1, input_dim)
                    else:
                        x_t = outputs[d - 1][i][j][0][0]
                        y_t = outputs[d - 1][i][j][1][0]

                    # ===================================== #
                    #    READ access to output array
                    # ===================================== #
                    states_x = output_d[i][j - 1][0]
                    states_y = output_d[i - 1][j][1]

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
                    output_d[i][j][0].append(h_x)
                    output_d[i][j][0].append(c_x)

                    # save hidden and cell state for direction y
                    output_d[i][j][1].append(h_y)
                    output_d[i][j][1].append(c_y)
            outputs.append(output_d)


if __name__ == "__main__":
    args = build_args_parser()

    for device in ["cpu", "cuda:0"]:
        cells = model_def(args.input_dim, args.hidden_dim, args.grid_dim,
                          args.depth, device)
        src_array_batch, trg_array_batch = gen_input_data(
            args.batch_size, args.input_dim, device=device)

        naive_grid_lstm([src_array_batch, trg_array_batch], args.depth,
                        args.batch_size, args.hidden_dim, cells, device)
