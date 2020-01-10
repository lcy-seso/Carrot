import pdb

import numpy as np
from time import time

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import *


def grid_lstm_skew_inner_3_loops(inputs, depth, batch_size, hidden_dim, cells,
                                 device):
    pass


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

        grid_lstm_skew_inner_3_loops([src_array_batch, trg_array_batch],
                                     args.depth, args.batch_size,
                                     args.hidden_dim, cells, device)
