import random
from typing import List

import torch
from torch import Tensor

__all__ = [
    'gen_sequence_batch',
]


def gen_sequence_batch(batch_size: int,
                       input_size: int,
                       min_len: int,
                       max_len: int,
                       device: str = 'cpu') -> List[Tensor]:
    """ Randomly generate one sequence batch on the device.

    Returns:
        A sequence batch, List[Tensor]. Each element of the returned list is a
        2D tensor with a shape of [sequence_length, input_dim].
    """
    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
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
    return batch_list, seq_len
