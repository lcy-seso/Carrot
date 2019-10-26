from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from .utils import Embedding, FineGrainedOpLSTMCell

layers = tf.keras.layers

__all__ = [
    'small_model',
]


class FineGrainedOpLSTMNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FineGrainedOpLSTMNet, self).__init__()

        self.hidden_size = hidden_size
        self.cells = [
            FineGrainedOpLSTMCell(input_size if i == 0 else hidden_size,
                                  hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def call(self, input_seq):
        batch_size = int(input_seq.shape[0])

        for rnncell in self.cells:  # iterate over depth
            outputs = []
            input_seq = tf.unstack(
                input_seq, num=int(input_seq.shape[1]), axis=1)
            h = tf.zeros((batch_size, self.hidden_size))
            c = tf.zeros((batch_size, self.hidden_size))
            for inp in input_seq:  # iterate over time step
                h, c = rnncell(inp, h, c)
                outputs.append(h)

            input_seq = tf.stack(outputs, axis=1)

        return [input_seq]


class StaticRNN(tf.keras.Model):
    """A static RNN.
    """

    def __init__(self, hidden_size, num_layers, use_cudnn_rnn=True):
        """
        hidden_size: Int, hidden dimension of the RNN unit.
        num_layers: Int, the number of stacked RNN unit, namely depth of the RNN
            network.
        """
        super(StaticRNN, self).__init__()

        if use_cudnn_rnn:
            self.cells = [
                tf.compat.v1.keras.layers.CuDNNLSTM(
                    hidden_size, return_state=True, return_sequences=True)
                for _ in range(num_layers)
            ]
        else:
            # About layers.LSTMCell's `implementation` argument, either 1 or 2.
            # Mode 1 will structure its operations as a larger number of smaller
            # dot products and additions, whereas mode 2 will batch them into
            # fewer, larger operations. These modes will have different
            # performance profiles on different hardware and for different
            # applications.
            self.cells = [
                layers.LSTMCell(
                    units=hidden_size,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros',
                    unit_forget_bias=True,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    implementation=2) for _ in range(num_layers)
            ]

        self.hidden_size = hidden_size
        self.use_cudnn_rnn = use_cudnn_rnn

    def _cudnn_lstm_call(self, input_seq):
        # A workaround to stack CuDNNLSTM in TF 2.0.
        # https://stackoverflow.com/questions/55324307/how-to-implement-a-stacked-rnns-in-tensorflow
        x = input_seq
        for rnn in self.cells:
            x = rnn(x)
        return x

    def call(self, input_seq):
        """Define computations in a single time step.

        input_seq: Tensor, the layout is
            [batch_size, max_sequence_length, embedding_dim].
        """
        if self.use_cudnn_rnn:
            return self._cudnn_lstm_call(input_seq)

        batch_size = int(input_seq.shape[0])

        for c in self.cells:  # iterate over depth
            state = (tf.zeros((batch_size, self.hidden_size)),
                     tf.zeros((batch_size, self.hidden_size)))
            outputs = []

            # unpack the input 3D tensors along the `max_sequence_length` axis
            # to get input tensors for each time step.
            input_seq = tf.unstack(
                input_seq, num=int(input_seq.shape[1]), axis=1)
            for inp in input_seq:  # iterate over time step
                output, state = c(inp, state)
                outputs.append(output)

            input_seq = tf.stack(outputs, axis=1)

        return [input_seq]


class PTBModel(tf.keras.Model):
    """LSTM for word language modeling.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 rnn_type):
        super(PTBModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.rnn_type = rnn_type
        self.embedding = Embedding(vocab_size, embedding_dim)

        if rnn_type == 'cudnn_lstm':
            self.rnn = StaticRNN(hidden_size, num_layers, True)
        elif rnn_type == 'static_lstm':
            self.rnn = StaticRNN(hidden_size, num_layers, False)
        elif rnn_type == 'fine_grained_op_lstm':
            self.rnn = FineGrainedOpLSTMNet(embedding_dim, hidden_size,
                                            num_layers)
        else:
            raise ValueError('Unknown RNN Type.')

        self.linear = layers.Dense(
            vocab_size,
            kernel_initializer=tf.keras.initializers.glorot_normal())
        self._output_shape = [-1, hidden_size]

    def call(self, input_seq):
        """Run the forward pass of PTBModel.
        Args:
            input_seq: Tensor(int64), layout: [batch_size, sequence_length].
        Returns:
            outputs tensors of inference with layout:
                [batch_size, sequence_length, hidden_size]
        """
        y = self.embedding(input_seq)
        if self.rnn is None:
            y = whileOpLSTM(y, self.hidden_size, self.hidden_size)
        else:
            y = self.rnn(y)[0]  # [batch_size, sequence_length, hidden_size]
        return self.linear(tf.reshape(y, self._output_shape))


def small_model(vocab_size, rnn_type):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=3,
        rnn_type=rnn_type)
