from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from utils import *


def __batch(xs: List[Tensor], dim=0) -> Tensor:
    batch_size = len(xs)
    return torch.reshape(torch.stack(xs, dim=dim), [batch_size, -1])


def __unbatch(x: Tensor, dim=0) -> List[Tensor]:
    return [a_slice.view(1, -1) for a_slice in torch.unbind(x, dim=dim)]


def grid_lstm_skew_inner_2_loops(
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
        for i in range(0, depth, 1):
            outputs.append(
                init_out_buff(src_length, trg_length, hidden_dim, device))

        for d in range(0, depth, 1):
            cell_x = cells[d][0]
            cell_y = cells[d][1]
            output_d = outputs[d]

            # ================================================================= #
            # Wavefront transformation to i loop and j loop.                    #
            # 2D transformation matrix for wavefront transformation is:         #
            #    [[1, 1]  * [i,   = [ i +j,                                     #
            #     [1, 0]]    j]         i  ]                                    #
            # After applying wavefront transformation, the outer i loop is      #
            # sequential, while the inner j loop is able to execute in parallel.#
            # ================================================================= #
            for i_proj in range(1 + 1, src_length + trg_length + 1, 1):

                x_t: List[Tensor] = []
                y_t: List[Tensor] = []
                states_x: List[List[Tensor], List[Tensor]] = []
                states_y: List[List[Tensor], List[Tensor]] = []

                gather_points = []
                for j_proj in range(
                        max(1, i_proj - trg_length), min(
                            src_length + 1, i_proj), 1):
                    # get index in the original iteration space.
                    i = j_proj
                    j = i_proj - j_proj
                    gather_points.append([i, j])

                    # ===================================== #
                    #            Gather inputs              #
                    # ===================================== #
                    if d == 0:
                        x_t.append(x[i - 1, :].view(1, input_dim))
                        y_t.append(y[j - 1, :].view(1, input_dim))
                    else:
                        x_t.append(outputs[d - 1][i][j][0][0])
                        y_t.append(outputs[d - 1][i][j][1][0])
                    states_x.append(output_d[i][j - 1][0])
                    states_y.append(output_d[i - 1][j][1])

                # ======================================================= #
                #   Batch inputs and perform cell computation             #
                # ======================================================= #
                x_t = __batch(x_t)
                y_t = __batch(y_t)

                h_x_prev = __batch([state[0] for state in states_x])
                c_x_prev = __batch([state[1] for state in states_x])
                h_y_prev = __batch([state[0] for state in states_y])
                c_y_prev = __batch([state[1] for state in states_y])

                h = torch.cat((h_x_prev, h_y_prev), dim=1)
                h_x, c_x = cell_x(x_t, (h, c_x_prev))
                h_y, c_y = cell_y(y_t, (h, c_y_prev))

                # ===================================== #
                #           Scatter outputs             #
                # ===================================== #
                h_x = __unbatch(h_x)
                c_x = __unbatch(c_x)
                h_y = __unbatch(h_y)
                c_y = __unbatch(c_y)

                for num, point in enumerate(gather_points):
                    output_d[point[0]][point[1]][0].append(h_x[num])
                    output_d[point[0]][point[1]][0].append(c_x[num])

                    output_d[point[0]][point[1]][1].append(h_y[num])
                    output_d[point[0]][point[1]][1].append(c_y[num])


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

        grid_lstm_skew_inner_2_loops(src_array_batch, trg_array_batch, cells,
                                     args.input_dim, args.hidden_dim, device)
