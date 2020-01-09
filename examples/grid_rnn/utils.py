import random
import argparse
import torch


def build_args_parser():
    parser = argparse.ArgumentParser(
        description="Compare different implementation of stacked LSTM")
    parser.add_argument("--batch_size", type=int, default=64)
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
        The input data for GridLSTM for NMT task, which is a list of 3-D Tensor.
    """
    src_array_batch = []
    trg_array_batch = []
    for i in range(batch_size):
        src_length = random.randint(MIN_LEN, MAX_LEN)
        trg_length = random.randint(MIN_LEN, MAX_LEN)

        # One input training examples is a 2-D tensor. The layout is:
        src_array_batch.append(
            torch.randn(src_length, input_dim, device=device))
        trg_array_batch.append(
            torch.randn(trg_length, input_dim, device=device))
    return src_array_batch, trg_array_batch
