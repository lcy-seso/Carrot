from typing import List, Tuple

import time
import random
import argparse
import torch
from torch import Tensor

from rnncell import LSTMCell

__all__ = [
    "gen_input_data",
    "model_def",
    "init_out_buff",
    "run_test",
    "__batch",
    "__unbatch",
]


def build_args_parser():
    parser = argparse.ArgumentParser(
        description="Compare different implementation of stacked LSTM")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--grid_dim", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--use_cuda", type=int, default=-1)
    return parser.parse_args()


def gen_input_data(batch_size, input_dim, device, MIN_LEN=80, MAX_LEN=100):
    """ Generate input data.

    Returns:
        Input sequence batch, List[Tensor].
        The input data for GridLSTM for NMT task, which is a list of 2-D Tensor.
    """
    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    src_array_batch = []
    trg_array_batch = []
    for i in range(batch_size):
        src_length = random.randint(MIN_LEN, MAX_LEN)
        trg_length = random.randint(MIN_LEN, MAX_LEN)

        # One examples is a 2-D tensor. The layout is: (seq_len, input_dim)
        src_array_batch.append(
            torch.randn(src_length, input_dim, device=device))
        trg_array_batch.append(
            torch.randn(trg_length, input_dim, device=device))
    return src_array_batch, trg_array_batch


def model_def(input_dim, hidden_dim, grid_dim, depth, device):
    return [[
        LSTMCell([input_dim, hidden_dim],
                 [hidden_dim * grid_dim, hidden_dim]).to(device),
        LSTMCell([input_dim, hidden_dim],
                 [hidden_dim * grid_dim, hidden_dim]).to(device)
    ] for d in range(depth)]


def init_out_buff(src_length, trg_length, hidden_dim, device):
    output_t = []
    for i in range(src_length + 1):
        output_t.append([[] for j in range(trg_length + 1)])

    for i in range(src_length + 1):
        for j in range(trg_length + 1):
            if i and j:
                state_x = []
                state_y = []
            else:
                state_x = zero_states(hidden_dim, device)
                state_y = zero_states(hidden_dim, device)

            output_t[i][j].append(state_x)
            output_t[i][j].append(state_y)
    return output_t


def zero_states(hidden_dim, device):
    return [
        torch.zeros(1, hidden_dim).to(device),
        torch.zeros(1, hidden_dim).to(device)
    ]


def __batch(xs: List[Tensor], dim=0) -> Tensor:
    batch_size = len(xs)
    return torch.reshape(torch.stack(xs, dim=dim), [batch_size, -1])


def __unbatch(x: Tensor, dim=0) -> List[Tensor]:
    return [a_slice.view(1, -1) for a_slice in torch.unbind(x, dim=dim)]


def run_test(model_func):
    args = build_args_parser()

    for device in [
            "cpu",
            "cuda:0",
    ]:
        cells = model_def(args.input_dim, args.hidden_dim, args.grid_dim,
                          args.depth, device)
        src_array_batch, trg_array_batch = gen_input_data(
            args.batch_size, args.input_dim, device=device)

        model_func(src_array_batch, trg_array_batch, cells, args.input_dim,
                   args.hidden_dim, device)
        print("finish warmup.")

        start = time.time()
        gather_time, compute_time, scatter_time = model_func(
            src_array_batch, trg_array_batch, cells, args.input_dim,
            args.hidden_dim, device)
        total = time.time() - start
        print("%s total time = %.4f" % (device, total))
        print("gather time = %.4f (%.2f%%)" % (gather_time, 100. *
                                               (gather_time / total)))
        print("compute time = %.4f (%.2f%%)" % (compute_time,
                                                100. * compute_time / total))
        print("scatter time = %.4f (%.2f%%)" % (scatter_time,
                                                100. * scatter_time / total))
        python_time = total - (gather_time + compute_time + scatter_time)
        print("other Python codes = %.4f(%.2f%%)\n" % (python_time, 100. *
                                                       (python_time / total)))
