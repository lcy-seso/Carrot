from typing import List, Tuple

import time
import random
import argparse
import torch
from torch import Tensor

__all__ = [
    'build_args_parser',
    'gen_input_data',
]


def build_args_parser():
    parser = argparse.ArgumentParser(
        description='Build the stacked clockwork rnn model.')
    parser.add_argument(
        '--batch_size', help='The batch size.', type=int, default=32)
    parser.add_argument(
        '--input_size', help='The input size.', type=int, default=32)
    parser.add_argument(
        '--block_size', help='The block size.', type=int, default=8)
    parser.add_argument(
        '--depth',
        help='the number of stacked RNN layer.',
        type=int,
        default=3)
    parser.add_argument(
        '--clock_periods',
        help='The clock period for CW-RNN cell. Must be delimited by \'.',
        type=str,
        default='1, 2, 4, 8')
    return parser.parse_args()


def gen_input_data(batch_size: int,
                   input_dim: int,
                   device: str,
                   MIN_LEN: int = 80,
                   MAX_LEN: int = 100) -> List[Tensor]:
    """ Generate random input data.

    Returns:
        A sequence batch, List[Tensor]. Each element of the returned list is a
        2D tensor with a shape of [sequence_length, input_dim].
    """
    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    return [
        torch.randn(
            random.randint(MIN_LEN,
                           MAX_LEN),  # Generate random sequence length
            input_dim,
            device=device) for _ in range(batch_size)
    ]
