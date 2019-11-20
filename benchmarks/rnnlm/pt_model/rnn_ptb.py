import pdb
from math import sqrt
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_ as init
from torch import Tensor

__all__ = [
    "small_model",
]


class FineGrainedOpLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(FineGrainedOpLSTMCell, self).__init__()
        # learnable paramters for input gate.
        self.Wi = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Ui = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bi = nn.Parameter(torch.ones(hidden_size))

        # learnable paramters for forget gate.
        self.Wf = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Uf = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bf = nn.Parameter(torch.ones(hidden_size))

        # learnable paramters for cell candidate.
        self.Wg = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Ug = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bg = nn.Parameter(torch.ones(hidden_size))

        # learnable paramters for output gate.
        self.Wo = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Uo = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bo = nn.Parameter(torch.ones(hidden_size))

    def forward(self, input: Tensor,
                state_prev: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward computation of a LSTM Cell.

        Args:
            input, Tensor, input to current time step with a shape of
                [batch_size, input_size].
            state_prev, Tuple, hidden states of previous time step.

        Returns:
            Hidden states of current time step.
        """

        h_prev, c_prev = state_prev

        ig: Tensor = torch.sigmoid(
            input.mm(self.Wi) + h_prev.mm(self.Ui) + self.bi)

        fg: Tensor = torch.sigmoid(
            input.mm(self.Wf) + h_prev.mm(self.Uf) + self.bf)

        c_candidate: Tensor = torch.tanh(
            input.mm(self.Wg) + h_prev.mm(self.Ug) + self.bg)

        og: Tensor = torch.sigmoid(
            input.mm(self.Wo) + h_prev.mm(self.Uo) + self.bo)

        c = fg.mul(c_prev) + ig.mul(c_candidate)
        h = og.mul(torch.tanh(c))
        return h, c


class StackedRNNNetJIT(nn.Module):
    def __init__(self, batch_size: int, max_seq_length: int, input_size: int,
                 hidden_size: int, num_layers: int, cell_type: str):
        super(StackedRNNNetJIT, self).__init__()

        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.register_buffer('init_state',
                             torch.zeros((batch_size, hidden_size)))

        if cell_type == "fine_grained_op_lstm":
            self.rnn1 = FineGrainedOpLSTMCell(input_size, hidden_size)
            self.rnn2 = FineGrainedOpLSTMCell(hidden_size, hidden_size)
            self.rnn3 = FineGrainedOpLSTMCell(hidden_size, hidden_size)
            self.rnn_cells = [self.rnn1, self.rnn2, self.rnn3]
        elif cell_type == "lstm_cell":
            self.rnn1 = torch.nn.LSTMCell(input_size, hidden_size)
            self.rnn2 = torch.nn.LSTMCell(hidden_size, hidden_size)
            self.rnn3 = torch.nn.LSTMCell(hidden_size, hidden_size)
            self.rnn_cells = [self.rnn1, self.rnn2, self.rnn3]
        else:
            raise ValueError("Unknown rnn type.")

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input, Tensor, input batch with a shape of [batch_size, seq_length,
                input_size].
        """
        xs = input

        h: Tensor = self.init_state
        c: Tensor = self.init_state

        hiddens: List[Tensor] = []
        cells: List[Tensor] = []
        for t in range(self.max_seq_length):
            x = xs[:, t, :]
            x, c = self.rnn1(x, (h, c))
            x, c = self.rnn2(x, (h, c))
            x, c = self.rnn3(x, (h, c))
            hiddens.append(h)
            cells.append(c)
        xs = torch.stack(hiddens, dim=1)
        return xs, torch.stack(cells)


class StackedRNNNet(nn.Module):
    def __init__(self, batch_size: int, max_seq_length: int, input_size: int,
                 hidden_size: int, num_layers: int, cell_type: str):
        super(StackedRNNNet, self).__init__()

        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.register_buffer('init_state',
                             torch.zeros((batch_size, hidden_size)))

        if cell_type == "fine_grained_op_lstm":
            self.rnn1 = FineGrainedOpLSTMCell(input_size, hidden_size)
            self.rnn2 = FineGrainedOpLSTMCell(hidden_size, hidden_size)
            self.rnn3 = FineGrainedOpLSTMCell(hidden_size, hidden_size)
            self.rnn_cells = [self.rnn1, self.rnn2, self.rnn3]
        elif cell_type == "lstm_cell":
            self.rnn1 = torch.nn.LSTMCell(input_size, hidden_size)
            self.rnn2 = torch.nn.LSTMCell(hidden_size, hidden_size)
            self.rnn3 = torch.nn.LSTMCell(hidden_size, hidden_size)
            self.rnn_cells = [self.rnn1, self.rnn2, self.rnn3]
        else:
            raise ValueError("Unknown rnn type.")

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input, Tensor, input batch with a shape of [batch_size, seq_length,
                input_size].
        """
        xs = input
        for rnn_cell in self.rnn_cells:
            h: Tensor = self.init_state
            c: Tensor = self.init_state

            hiddens: List[Tensor] = []
            cells: List[Tensor] = []
            for t in range(self.max_seq_length):
                x = xs[:, t, :]
                h, c = rnn_cell(x, (h, c))
                hiddens.append(h)
                cells.append(c)
            xs = torch.stack(hiddens, dim=1)
        return xs, torch.stack(cells)


class CuDNNLSTMPTBModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int,
                 rnn_hidden_dim: int, num_layers: int):
        super(CuDNNLSTMPTBModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # The model uses the nn.RNN module (and its sister modules nn.GRU
        # and nn.LSTM) which will automatically use the cuDNN backend
        # if run on CUDA with cuDNN installed.
        self.rnn_net = nn.LSTM(
            embedding_dim,
            rnn_hidden_dim,
            num_layers,
            dropout=0.,
            batch_first=True,
            bidirectional=False)
        self.linear = nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, input):
        embedding = self.embedding(input)
        hiddens, _ = self.rnn_net(embedding)
        return self.linear(hiddens), hiddens


class PTBModel(nn.Module):
    """Container module for a stacked RNN language model.

    Args:
        cell_type: str, the recurrent net to compute sentence representation.
        vocab_size: int, the size of word vocabulary.
        embedding_dim: int, the dimension of word embedding.
        rnn_hidden_dim: int, the dimension of RNN cell's hidden state.
        num_layers: int, the number of stacked RNN network.
        enable_jit: bool, The stacked RNN language model is a nested loop net.
            In current implementation, the outer loop iterates over depth and
            the inner loop iterates over sequence length. This flag determines
            whether to apply PyTorch JIT to the outer loop.
    Returns:
        A Tuple of Tensor. The first element is the final output of the
        model before loss computation with a shape of
            [batch_size, seq_len, vocab_size.
        The second element is the hidden states of the RNN network with a shape
            [batch_size, seq_len, rnn_hidden_dim].
    """

    def __init__(self, batch_size: int, max_seq_length: int, cell_type: str,
                 vocab_size: int, embedding_dim: int, rnn_hidden_dim: int,
                 num_layers: int, enable_jit: bool):
        super(PTBModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if enable_jit:
            self.rnn_net = torch.jit.script(
                StackedRNNNetJIT(batch_size, max_seq_length, embedding_dim,
                                 rnn_hidden_dim, num_layers, cell_type))
            # print(self.rnn_net.graph)
        else:
            self.rnn_net = StackedRNNNet(batch_size, max_seq_length,
                                         embedding_dim, rnn_hidden_dim,
                                         num_layers, cell_type)

        self.linear = nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, input):
        """Define forward computations of the RNNLM.

        Args:
            input: Tensor, its shape is [bath_size, seq_len].

        Returns:
            A Tensor with a shape of [batch_size, seq_len, rnn_hidden_dim].
        """

        def _model_func(x):
            # layout of `embedding_dim`: [batch_size, seq_len, embedding_dim]
            embedding = self.embedding(x)
            hiddens, _ = self.rnn_net(embedding)
            return self.linear(hiddens), hiddens

        return _model_func(input)


def small_model(cell_type,
                batch_size,
                max_seq_length,
                vocab_size,
                enable_jit=False):
    if cell_type == "cudnn_lstm":
        return CuDNNLSTMPTBModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            rnn_hidden_dim=256,
            num_layers=3)
    else:
        return PTBModel(
            cell_type=cell_type,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            vocab_size=vocab_size,
            embedding_dim=128,
            rnn_hidden_dim=256,
            num_layers=3,
            enable_jit=enable_jit)
