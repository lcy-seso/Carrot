from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .utils import Embedding

layers = tf.keras.layers

__all__ = [
    'small_model',
]


class StaticRNN(tf.keras.Model):
    """A static RNN.
    """

    def __init__(self, hidden_dim, num_layers, use_cudnn_rnn=True):
        """
        hidden_dim: Int, hidden dimension of the RNN unit.
        num_layers: Int, the number of stacked RNN unit, namely depth of the RNN
            network.
        """
        super(StaticRNN, self).__init__()

        if use_cudnn_rnn:
            self.cells = [
                tf.compat.v1.keras.layers.CuDNNLSTM(
                    hidden_dim, return_state=True, return_sequences=True)
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
                    units=hidden_dim,
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

        self.hidden_dim = hidden_dim
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

        # NOTE: below line only works in eager mode.
        # In eager mode, TensorFlow operations are immediately evaluated and
        # return their values to Python, so below line will return the vaule
        # of batch size, while in graph mode, we use dataset API to feed data,
        # the batch size dimension is None when defining the graph.
        batch_size = int(input_seq.shape[0])

        for c in self.cells:  # iterate over depth
            state = (tf.zeros((batch_size, self.hidden_dim)),
                     tf.zeros((batch_size, self.hidden_dim)))
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

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 rnn_type):
        super(PTBModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.rnn_type = rnn_type
        self.embedding = Embedding(vocab_size, embedding_dim)

        if rnn_type == 'cudnn_lstm':
            self.rnn = StaticRNN(hidden_dim, num_layers, True)
        elif rnn_type == 'static_lstm':
            self.rnn = StaticRNN(hidden_dim, num_layers, False)
        elif rnn_type == 'fine_grained_lstm_eager':
            raise Exception('Not impelmented yet.')
        else:
            raise ValueError('Unknown RNN Type.')

        self.linear = layers.Dense(
            vocab_size,
            kernel_initializer=tf.keras.initializers.glorot_normal())
        self._output_shape = [-1, hidden_dim]

    def call(self, input_seq):
        """Run the forward pass of PTBModel.
        Args:
            input_seq: Tensor(int64), layout: [batch_size, sequence_length].
        Returns:
            outputs tensors of inference with layout:
                [batch_size, sequence_length, hidden_dim]
        """
        y = self.embedding(input_seq)
        if self.rnn is None:
            y = whileOpLSTM(y, self.hidden_dim, self.hidden_dim)
        else:
            y = self.rnn(y)[0]  # [batch_size, sequence_length, hidden_dim]
        return self.linear(tf.reshape(y, self._output_shape))


def small_model(vocab_size, rnn_type):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=3,
        rnn_type=rnn_type)
