"""Forward computation of the Grid Long Short Term Memory network for NMT.

Please refer to the paper 'Kalchbrenner, Nal, Ivo Danihelka, and Alex Graves.
Grid long short-term memory. arXiv preprint arXiv:1507.01526 (2015).' for details.
"""
import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functools import reduce
import random
from time import time
from typing import List, Tuple
from itertools import groupby
from collections import namedtuple

import torch
from torch import Tensor
from torch.nn import Module

from data_utils import build_args_parser
from utils import VanillaRNNCell_


class IV(namedtuple('IterationVector', (
        'orig',
        'T',
))):
    pass


# Transformation matrix.
T = np.array(
    [
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.int32)
# Original dependence vectors.
deps = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.int32)
# Transformed dependences.
deps_trans = T @ deps

DEP_NUM = deps.shape[1]
DEPS = [deps[:, i].tolist() for i in range(DEP_NUM)]
DEPS_T = [deps_trans[:, i].tolist() for i in range(DEP_NUM)]


class WriteBuff(object):
    def __init__(self, *size, device) -> Tensor:
        if len(size) != 2:
            raise NotImplemented('Buffer other than 2D is not supported.')

        self.column_size = size[1]
        self.T = torch.zeros(size, device=device)
        # default dtype = torch.float32
        print("output buffer size = %.4f GB" %
              (reduce(lambda x, y: x * y, size) * 4 / 1024 / 1024 / 1024))

        # write from the second row.
        self._write_pointer = 1
        self._address = {}

    def to_physical_addr(self, z, m, n, p, is_y_direction=False) -> int:
        return (self._address[f'y#{z}#{m}#{n}#{p}']
                if is_y_direction else self._address[f'{z}#{m}#{n}#{p}'])

    def get_write_tensor(self, iv: List[int]) -> Tensor:
        length = len(iv)
        tensor = torch.as_strided(
            self.T,
            size=(length, self.column_size),
            stride=(self.column_size, 1),
            storage_offset=self._write_pointer *
            self.column_size)  # the first row is the init state.

        for i, v in enumerate(iv):
            key = '#'.join(map(str, v))
            if key in self._address:
                # FIXME(Ying): Key is hardcoded. In an iteration,
                # the output buffer is written twice by computations from two
                # directions. The prefix 'y#' is to make the key unique.
                self._address['y#' + key] = self._write_pointer + i
            else:
                self._address[key] = self._write_pointer + i
        self._write_pointer += length
        return tensor


class GridRNNSkew(Module):
    """For experiment only.

    The cell computation uses inplace tensor operation. It CANNOT be
    automatically differentiated in PT's default autograd implementation,
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 grid_dim: int,
                 depth: int,
                 device: str = 'cpu'):
        super(GridRNNSkew, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grid_dim = grid_dim
        self.depth = depth
        self.cells_x = [
            VanillaRNNCell_(
                input_size, hidden_size, device, grid_dim=grid_dim)
            for i in range(depth)
        ]
        self.cells_y = [
            VanillaRNNCell_(
                input_size, hidden_size, device, grid_dim=grid_dim)
            for i in range(depth)
        ]
        self.device = device

    def get_deps(self, p, dvs):
        """
        Given a point p and all dependence vectors, returns all
        p's precursors (dependences).
        """
        return [[x - y for x, y in zip(p, d)] for d in dvs]

    def __batch(self, xs: Tensor, index: List[int], dim=0) -> Tensor:
        return torch.index_select(xs, dim, torch.tensor(
            index, device=xs.device))

    def __unbatch(self, x: Tensor, dim=0) -> List[Tensor]:
        chunks = x.size()[0]
        return torch.chunk(x, chunks, dim=0)

    def __len_to_start_pos(self, lens):
        start_pos = [0]
        for i, l in enumerate(lens):
            start_pos.append(start_pos[i] + lens[i])
        return start_pos

    def forward(self, src_array_batch: List[Tensor],
                trg_array_batch: List[Tensor], src_lens: List[int],
                trg_lens: List[int]):
        src_start_pos = self.__len_to_start_pos(src_lens)
        trg_start_pos = self.__len_to_start_pos(trg_lens)

        batch_size = len(src_lens)
        row_size = self.depth * sum([
            src_len * trg_len * self.grid_dim
            for src_len, trg_len in zip(src_lens, trg_lens)
        ]) + 1  # the first row is preserved for the zero initial state.
        outputs = WriteBuff(row_size, self.hidden_size, device=self.device)

        z_upper = max(
            [s + t + self.depth - 2 for s, t in zip(src_lens, trg_lens)])
        for z in range(0, z_upper, 1):
            # NOTE: m loop is also a parallel loop, but current implementation
            # dose not batch computation graphs, and leaves it sequential.
            for m in range(0, self.depth):
                parallel_points = []

                # parallel loops
                for n in range(0, batch_size, 1):
                    for p in range(0, src_lens[n], 1):
                        d = m
                        sample_id = n
                        i = p
                        j = z - d - i

                        if j < 0 or j >= trg_lens[sample_id]: continue

                        parallel_points.append(
                            IV(orig=[sample_id, d, i, j], T=[z, m, n, p]))

                if not parallel_points: continue

                xt_indices = []
                yt_indices = []
                x_state_indices = []
                y_state_indices = []
                for point in parallel_points:
                    z, d, sample_id, i = point.T
                    _, _, _, j = point.orig

                    if d == 0:  # depth = 0
                        # access `src_array_batch` and `trg_array_batch` to
                        # get inputs.
                        xt_indices.append(src_start_pos[sample_id] + i)
                        yt_indices.append(trg_start_pos[sample_id] + j)
                    else:
                        # access `outputs` to get inputs.
                        xt_indices.append(
                            outputs.to_physical_addr(sample_id, d - 1, i, j))
                        yt_indices.append(
                            outputs.to_physical_addr(sample_id, d - 1, i, j,
                                                     True))

                    if i == 0:
                        x_state_indices.append(0)  # initial states
                    else:
                        addr = outputs.to_physical_addr(sample_id, d, i - 1, j)
                        x_state_indices.append(addr)

                    if j == 0:
                        y_state_indices.append(0)  # initial states
                    else:
                        addr = outputs.to_physical_addr(
                            sample_id, d, i, j - 1, True)
                        y_state_indices.append(addr)

                # batch data
                x_t = self.__batch(src_array_batch
                                   if d == 0 else outputs.T, xt_indices)
                y_t = self.__batch(trg_array_batch
                                   if d == 0 else outputs.T, yt_indices)
                state_x = self.__batch(outputs.T, x_state_indices)
                state_y = self.__batch(outputs.T, y_state_indices)

                # compute
                h_prev = torch.cat((state_x, state_y), dim=1)

                out = outputs.get_write_tensor(
                    [p.orig for p in parallel_points])
                self.cells_x[d](out, x_t, h_prev)

                out = outputs.get_write_tensor(
                    [p.orig for p in parallel_points])
                self.cells_y[d](out, y_t, h_prev)

        return outputs


def test_skew(args, grid_dim, device):
    random.seed(1020)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    def gen_seq_batch(batch_size: int, input_size: int, min_len: int,
                      max_len: int, device: str) -> Tuple[Tensor, List[int]]:
        lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
        batch = torch.randn((sum(lens), input_size), device=device)
        return batch, lens

    min_len = args.min_len
    max_len = args.max_len
    batch_size = args.batch_size
    src_array_batch, src_lens = gen_seq_batch(batch_size, args.input_size,
                                              min_len, max_len, device)
    trg_array_batch, trg_lens = gen_seq_batch(batch_size, args.input_size,
                                              min_len, max_len, device)

    m = GridRNNSkew(args.input_size, args.hidden_size, grid_dim, args.depth,
                    device)

    start = time()
    m(src_array_batch=src_array_batch,
      trg_array_batch=trg_array_batch,
      src_lens=src_lens,
      trg_lens=trg_lens)
    print('Time elapse = %.6f (s).' % (time() - start))
    print("Finish warmup. Start timing.")

    start = time()
    m(src_array_batch=src_array_batch,
      trg_array_batch=trg_array_batch,
      src_lens=src_lens,
      trg_lens=trg_lens)
    print('Time elapse = %.6f (s).' % (time() - start))


if __name__ == "__main__":
    args = build_args_parser()
    grid_dim = 2  # Current implementation fixes the grid dim to be 2.

    for device in [
            'cpu',
            'cuda',
    ]:

        print('\n---------------------------------------------------------')
        print(f'Run test on {device}.')
        test_skew(args, grid_dim, device)
