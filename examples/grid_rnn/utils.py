import random
import argparse
import torch

from rnncell import LSTMCell

__all__ = [
    "build_args_parser",
    "gen_input_data",
    "model_def",
    "init_out_buff",
]


def build_args_parser():
    parser = argparse.ArgumentParser(
        description="Compare different implementation of stacked LSTM")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=2)
    parser.add_argument("--grid_dim", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--use_cuda", type=int, default=-1)
    return parser.parse_args()


def gen_input_data(batch_size, input_dim, device, MIN_LEN=5, MAX_LEN=25):
    """ Generate input data.

    Returns:
        Input sequence batch, List[Tensor].
        The input data for GridLSTM for NMT task, which is a list of 2-D Tensor.
    """
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
