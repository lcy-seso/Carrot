import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import random
import torch
from torch import Tensor

from utils import VanillaRNNCell


class StackedLSTM(torch.jit.ScriptModule):
    def __init__(self, hidden_size, cells):
        super(StackedLSTM, self).__init__()
        self.cells = torch.nn.ModuleList(cells)
        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

    @torch.jit.script_method
    def forward(
            self,
            input_seq: List[List[Tensor]],  # Input data
            lens: List[int],
            batch_size: int,
            depth: int):
        """
        Argument 1 and Argument 2 are read-only input arrays.

        To make a program description independent of the size of data,
        parameters 3 to 5 are introduced which stand for possible size.
        """
        output_i: List[List[List[Tensor]]] = []
        for i in range(batch_size):  # parallelizable loop

            output_j: List[List[Tensor]] = []
            for d, cell in enumerate(self.cells):

                output_k: List[Tensor] = []
                for t in range(lens[i]):  # iterate over variable length data
                    if d == 0:
                        x = input_seq[i][t]
                    else:
                        x = output_j[d - 1][t]

                    if t == 0:
                        h_prev = self.init_state
                    else:
                        h_prev = output_k[-1]

                    h = cell(x, h_prev)
                    output_k.append(h)
                output_j.append(output_k)
            output_i.append(output_j)
        return output_i


def run_test(seq_batch,
             lens,
             batch_size,
             depth,
             input_size,
             hidden_size,
             device='cuda'):
    cells = [
        VanillaRNNCell(input_size, hidden_size).to(device)
        for _ in range(depth)
    ]
    m = StackedLSTM(hidden_size, cells).to(device)
    print(torch.jit.script(m).graph)

    m(seq_batch, lens, batch_size, depth)


if __name__ == '__main__':
    batch_size = 13
    depth = 5
    min_len = 5
    max_len = 15

    input_size = 32
    hidden_size = 32

    device = 'cuda'

    random.seed(1234)
    torch.manual_seed(1234)

    seq_batch = []
    lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
    for i in range(batch_size):
        seq_batch.append([])
        for j in range(lens[i]):
            seq_batch[-1].append(torch.randn(1, input_size, device=device))

    run_test(seq_batch, lens, batch_size, depth, input_size, hidden_size,
             device)
