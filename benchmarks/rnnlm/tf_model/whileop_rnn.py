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


class WhileOpLSTMNet(tf.keras.Model):
    """LSTM implemented in fine-grained operators via symbolic while-ops.

    Only works in graph-mode.
    Args:
        input (Tensor): [batch_size, seq_len, input_dim]
        hidden_size(int): hidden/cell dimension
        output_size(int): output dimension

    Return:
        A Tensor with a shape [batch_size, sequence_length, hidden_dim]
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(WhileOpLSTMNet, self).__init__()

        self.rnncells = [
            FineGrainedOpLSTMCell(input_size, hidden_size, output_size)
            for _ in range(num_layers)
        ]
        self.hidden_size = hidden_size

    def _while_op_lstm(self, input):
        shape = tf.shape(input)
        batch_size = shape[0]
        seq_len = shape[1]
        input_size = self.input_size

        init_state = (tf.zeros([batch_size, self.hidden_size]),
                      tf.zeros([batch_size, self.hidden_size]))

        init_t = tf.constant(0)
        init_output_array = tf.TensorArray(dtype=tf.float32, size=seq_len)
        cond = lambda i, _: tf.less(i, seq_len)

        def body(t, step):
            states_prev, output_array = step

            x = input[:, t]
            states = []
            for state_prev, rnncell in zip(states_prev, self.rnncells):
                h, c = rnncell(x, state_prev)
                x = h
                states.append((h, c))

            return t + 1, (states, output_array.write(t, x))

        _, step = tf.while_loop(cond, body, (init_t,
                                             (init_state, init_output_array)))
        _, _, output_array = step
        return output_array.stack()

    def call(self, input_seq):
        """Stacked LSTM network implemented by symbolic operators.

        Args:
            input_seq, Tensor, input sequence batch. The layout must be
                batch_size major: [batch_size, seq_len, input_dim].
        """
        return self._while_op_lstm(input_seq)


class PTBModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(PTBModel, self).__init__()

        self.emb = Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim

        self.linear = layers.Dense(
            vocab_size,
            kernel_initializer=tf.keras.initializers.glorot_normal())
        self._output_shape = [-1, hidden_dim]

        self.rnn = WhileOpLSTMNet(embedding_dim, hidden_dim, hidden_dim,
                                  num_layers)

    def call(self, input):
        x = self.emb(input)
        x = self.rnn(x)
        return self.linear(tf.reshape(x, self._output_shape))


def small_model(vocab_size):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=3)
