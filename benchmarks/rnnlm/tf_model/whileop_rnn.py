from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from .utils import Embedding

layers = tf.keras.layers

__all__ = [
    'small_model',
]


def whileOpLSTM(input, hidden_size, output_size):
    """LSTM implemented in fine-grained operators via symbolic while-ops.

    Only workd in graph-mode.
    Args:
        input (Tensor): [batch_size, seq_len, input_dim]
        hidden_size(int): hidden/cell dimension
        output_size(int): output dimension

    Return:
        A Tensor with a shape [batch_size, sequence_length, hidden_dim]
    """

    shape = tf.shape(input)
    batch_size = shape[0]
    seq_len = shape[1]
    input_size = shape[2]

    stddev = 1.0 / math.sqrt(hidden_size)

    with tf.name_scope('gate'):
        # Concatenate input-to-hidden and hidden-to-hidden projections
        # into one projection.
        w_gate = tf.Variable(
            tf.random.uniform(
                [input_size + hidden_size, hidden_size],
                minval=-stddev,
                maxval=stddev))  # [input_size + hidden_size, hidden_size]

        # gate bias: [hidden_size]
        b_gate = tf.Variable(
            tf.random.uniform([hidden_size], minval=-stddev, maxval=stddev))

    def gate(input):
        """Gate projection without activation.
            Args:
                input (Tensor), layout [batch_size, input_size + hidden_size].

            Returns:
                A Tensor with a shape [batch_size, hidden_size].
            """
        return tf.matmul(input, w_gate) + b_gate

    with tf.name_scope('output'):
        # hidden-to-output projection.
        w_output = tf.Variable(
            tf.random.uniform(
                [hidden_size, output_size], minval=-stddev, maxval=stddev))

        # output bias: [output_size]
        b_output = tf.Variable(
            tf.random.uniform([output_size], minval=-stddev, maxval=stddev))

    def output(input):
        """hidden-to-output projection.
            Args:
                input (Tensor), layout [batch_size, input_size + hidden_size]

            Returns:
                A Tensor with a shape [batch_size, output_size]
            """
        return tf.matmul(input, w_output) + b_output

    init_hidden = tf.zeros([batch_size,
                            hidden_size])  # [batch_size, hidden_size]

    init_cell = tf.zeros([batch_size, hidden_size])
    init_i = tf.constant(0)
    init_output_array = tf.TensorArray(dtype=tf.float32, size=seq_len)
    cond = lambda i, _: tf.less(i, seq_len)

    def body(i, step):
        """LSTM cell.
            """
        hidden, cell, output_array = step

        x_t = input[:, i]

        combined = tf.concat([x_t, hidden], 1)

        f_gate = gate(combined)
        i_gate = gate(combined)
        o_gate = gate(combined)

        f_gate = tf.math.sigmoid(f_gate)
        i_gate = tf.math.sigmoid(i_gate)
        o_gate = tf.math.sigmoid(o_gate)

        cell = tf.math.add(
            tf.math.multiply(cell, f_gate),
            tf.math.multiply(tf.math.tanh(gate(combined)), i_gate))

        hidden = tf.math.multiply(tf.math.tanh(cell), o_gate)

        output_value = output(hidden)

        return i + 1, (hidden, cell, output_array.write(i, output_value))

    _, step = tf.while_loop(
        cond, body, (init_i, (init_hidden, init_cell, init_output_array)))
    _, _, output_array = step

    return output_array.stack()


class PTBModel(object):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        self.emb = Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim

        self.linear = layers.Dense(
            vocab_size,
            kernel_initializer=tf.keras.initializers.glorot_normal())
        self._output_shape = [-1, hidden_dim]

    def __call__(self, input):
        x = self.emb(input)
        x = whileOpLSTM(x, self.hidden_dim, self.hidden_dim)
        return self.linear(tf.reshape(x, self._output_shape))


def small_model(input, vocab_size):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=3)
