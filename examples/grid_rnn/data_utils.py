from typing import List, Tuple
import random
import argparse

import torch
from torch import Tensor

__all__ = [
    'build_args_parser',
    'gen_contiguous_input_data',
]


def build_args_parser():
    parser = argparse.ArgumentParser(
        description='Build the Grid LSTM for NMT.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=260,
        help=('The batch size defines the number of '
              'samples that will be propagated through the network.'))
    parser.add_argument(
        '--input_size',
        type=int,
        default=200,
        help="The number of neurons in GridRNN's input layer.")
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=200,
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


def gen_one_batch(batch_size, input_size, min_len, max_len, device):
    seq_len = [random.randint(min_len, max_len) for _ in range(batch_size)]
    batch = torch.randn(sum(seq_len), input_size, device=device)

    offset = 0
    batch_list = []
    for i in range(batch_size):
        a_seq = torch.as_strided(
            batch,
            size=(seq_len[i], input_size),
            stride=(input_size, 1),
            storage_offset=offset)
        offset += seq_len[i] * input_size
        batch_list.append(a_seq)
    return batch_list


def gen_contiguous_input_data(batch_size, input_size, min_len, max_len,
                              device):
    """Generate input data.

    Returns:
        Input sequence batch, List[Tensor].
        The input data for GridLSTM for NMT task, which is a list of 2-D Tensor.
    """
    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    src_array_batch = gen_one_batch(batch_size, input_size, min_len, max_len,
                                    device)
    trg_array_batch = gen_one_batch(batch_size, input_size, min_len, max_len,
                                    device)
    return src_array_batch, trg_array_batch
