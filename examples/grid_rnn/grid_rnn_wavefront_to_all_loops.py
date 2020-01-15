import pdb

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
    # batch_size and depth are constants independent of data.
    batch_size = len(src_array_batch)
    depth = len(cells)

    # ===================================== #
    #    Initialize output buffer           #
    # ===================================== #
    src_lens = [src_array_batch[i].size()[0] for i in range(batch_size)]
    trg_lens = [trg_array_batch[i].size()[0] for i in range(batch_size)]
    max_src_len = max(src_lens)
    max_trg_len = max(trg_lens)

    # `outputs` is the output buffer. A nested array with a depth 4 is used.
    # depth 1: for each sample;
    # depth 2: for each depth of the neural network;
    # depth 3: for each direction;
    # depth 4: for hidden states and cells;
    point_count = 0
    outputs: List[List[List[List[Tensor]]]] = []
    for src_length, trg_length in zip(src_lens, trg_lens):
        point_count += src_length * trg_length * depth

        outputs.append([])
        for i in range(depth):
            outputs[-1].append(
                init_out_buff(src_length, trg_length, hidden_dim, device))
    print("total data points = %d" % (point_count))

    # ===================================================================== #
    # Wavefront transformation to the entire loop nesting.
    # 4D transformation matrix for the wavefront transformation is:
    #    [[1, 1, 1, 1]
    #     [1, 0, 0, 0]
    #     [0, 1, 0, 0]
    #     [0, 0, 1, 0]]
    # After applying wavefront transformation, the uttermost `m` loop is
    # sequential, while all the inner loops are able to execute in parallel.
    # This is still NOT the optimal parallelisms.
    # ===================================================================== #
    gather_points = []

    for m in range(0 + 0 + 1 + 1,
                   batch_size + depth + max_src_len + max_trg_len + 1, 1):

        n_low = max(1, m - depth - max_src_len)
        n_high = min(batch_size, m)
        print("n = [%d to %d]" % (n_low, n_high))
        for n in range(max(0, m - depth - max_src_len), min(batch_size, m), 1):

            p_low = max(1, n - max_src_len)
            p_high = min(depth, n)
            print("n = [%d to %d]" % (p_low, p_high))
            for p in range(p_low, p_high, 1):  # depth

                q_low = max(1, p - max_trg_len)
                q_high = min(max_src_len + 1, p)
                print("n = [%d to %d]" % (q_low, q_high))
                for q in range(q_low, q_high, 1):  # src_sequence
                    # This is the implementation of a schedule function.
                    # Since the memory acesses satisfy affine constrains,
                    # the schedule function is possible to be determined by
                    # solving certain linear programming problem.
                    # get coordinate values in the original iteration space.
                    sample_id = m - n - p - q
                    d = n
                    i = p
                    j = q
                    if i > src_lens[sample_id] or j > trg_lens[sample_id]:
                        continue
                    gather_points.append([sample_id, d, i, j])


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

        grid_lstm_skew_to_outermost_loop(src_array_batch, trg_array_batch,
                                         cells, args.input_dim,
                                         args.hidden_dim, device)
