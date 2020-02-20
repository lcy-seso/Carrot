import pdb

from typing import List, Tuple
from time import time
import random
import argparse

import torch
from torch import Tensor

from rnncell import LSTMCell

__all__ = [
    'build_args_parser',
    'gen_input_data',
    'gen_contiguous_input_data',
    'model_def',
    'init_out_buff',
    'run_test',
    '__batch',
    '__unbatch',
]


def build_args_parser():
    parser = argparse.ArgumentParser(
        description='Build the Grid LSTM for NMT.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help=('The batch size defines the number of '
              'samples that will be propagated through the network.'))
    parser.add_argument(
        '--input_dim',
        type=int,
        default=64,
        help="The number of neurons in GridRNN's input layer.")
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help="The number of neurons in GridRNN's hidden layer.")
    parser.add_argument(
        '--depth',
        type=int,
        default=3,
        help='The number of stacked RNN layer.')
    parser.add_argument(
        '--min_len', type=int, default=80, help='The minimum sequence length.')
    parser.add_argument(
        '--max_len',
        type=int,
        default=100,
        help='The maximum sequence length.')
    return parser.parse_args()


def gen_one_batch(batch_size, input_dim, min_len, max_len, device):
    seq_len = [random.randint(min_len, max_len) for _ in range(batch_size)]
    batch = torch.randn(sum(seq_len), input_dim, device=device)

    offset = 0
    batch_list = []
    for i in range(batch_size):
        a_seq = torch.as_strided(
            batch,
            size=(seq_len[i], input_dim),
            stride=(input_dim, 1),
            storage_offset=offset)
        offset += seq_len[i] * input_dim
        batch_list.append(a_seq)
    return batch_list


def gen_contiguous_input_data(batch_size, input_dim, min_len, max_len, device):
    """Generate input data.

    Returns:
        Input sequence batch, List[Tensor].
        The input data for GridLSTM for NMT task, which is a list of 2-D Tensor.
    """
    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    src_array_batch = gen_one_batch(batch_size, input_dim, min_len, max_len,
                                    device)
    trg_array_batch = gen_one_batch(batch_size, input_dim, min_len, max_len,
                                    device)
    return src_array_batch, trg_array_batch


def gen_input_data(batch_size, input_dim, device, min_len, max_len):
    """Generate input data.

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
        src_length = random.randint(min_len, max_len)
        trg_length = random.randint(min_len, max_len)

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
    return torch.cat(xs, dim=dim)


def __unbatch(x: Tensor, dim=0) -> List[Tensor]:
    chunks = x.size()[0]
    return torch.chunk(x, chunks, dim=0)


def run_test(model_func):
    args = build_args_parser()
    grid_dim = 2  # Current implementation fixes the grid dim to be 2.

    for device in [
            # 'cpu',
            'cuda',
    ]:

        print('\n---------------------------------------------------------')
        print(f'Run test on {device}.')

        cells = model_def(args.input_dim, args.hidden_dim, grid_dim,
                          args.depth, device)
        src_array_batch, trg_array_batch = gen_contiguous_input_data(
            args.batch_size,
            args.input_dim,
            args.min_len,
            args.max_len,
            device=device)
        model_func(src_array_batch, trg_array_batch, cells, args.input_dim,
                   args.hidden_dim, device)
        print("Finish warmup. Start timing.")

        start = time()
        model_func(src_array_batch, trg_array_batch, cells, args.input_dim,
                   args.hidden_dim, device)
        print("Time elapse = %.6f" % (time() - start))

        # with torch.autograd.profiler.profile(
        #         enabled=True,
        #         use_cuda=False if device == "cpu" else True,
        #         record_shapes=False) as prof:
        #     # warmup GPU allocator.
        #     model_func(src_array_batch, trg_array_batch, cells, args.input_dim,
        #                args.hidden_dim, device)
        #     print("Finish warmup. Start timing.")

        #     start = time()
        #     model_func(src_array_batch, trg_array_batch, cells, args.input_dim,
        #                args.hidden_dim, device)
        #     print("Time elapse = %.6f" % (time() - start))
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))


if __name__ == "__main__":
    gen_contiguous_input_data(batch_size=16, input_dim=32, device="cuda:0")
