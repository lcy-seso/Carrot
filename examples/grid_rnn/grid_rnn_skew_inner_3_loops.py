import pdb

from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

from utils import *


def grid_lstm_skew_inner_3_loops(
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

    # ===================================== #
    #    Initialize output buffer           #
    # ===================================== #
    outputs = []
    for i in range(depth):
        outputs.append(
            init_out_buff(src_length, trg_length, hidden_dim, device))

    # data parallelism: iterate over samples in a batch.
    for d in range(0, depth, 1):
        cell_x = cells[d][0]
        cell_y = cells[d][1]

        output_d = outputs[d]

        src_lens = [src_array_batch]
        for sample_id in range(0, batch_size, 1):
            for i in range(0, new_loop_boundary, 1):
                for j in range():
                    x = src_array_batch[sample_id]
                    y = trg_array_batch[sample_id]


if __name__ == "__main__":
    args = build_args_parser()

    for device in [
            "cpu",  #
            # "cuda:0",
    ]:
        cells = model_def(args.input_dim, args.hidden_dim, args.grid_dim,
                          args.depth, device)
        src_array_batch, trg_array_batch = gen_input_data(
            args.batch_size, args.input_dim, device=device)

        grid_lstm_skew_inner_3_loops(src_array_batch, trg_array_batch, cells,
                                     args.input_dim, args.hidden_dim, device)
