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


class WhileOpLSTM(tf.keras.Model):
    """LSTM implemented in fine-grained operators via symbolic while-ops.

    Only works in graph-mode.
    Args:
        input (Tensor): [batch_size, seq_len, input_dim]
        hidden_size(int): hidden/cell dimension
        output_size(int): output dimension

    Return:
        A Tensor with a shape [batch_size, sequence_length, hidden_dim]
    """

    def __init__(self, hiddens, output_size):
        super(WhileOpLSTM, self).__init__()

        self.hiddens = hiddens
        self._i2h_shapes = [[h, h] for h in hiddens]
        self._h2h_shapes = [[h, h] for h in hiddens]
        self._h2o_shapes = [[h, h] for h in hiddens]
        self._h2o_shapes[-1][-1] = output_size
        self.num_layers = len(hiddens)

    def _gate(self, input, w, b):
        """Gate projection without activation.
        Args:
            input (Tensor), layout [batch_size, input_size + hidden_size].

        Returns:
            A Tensor with a shape [batch_size, hidden_size].
        """
        return tf.matmul(input, w) + b

    def _output(self, input, w, b):
        """hidden-to-output projection.
        Args:
            input (Tensor), layout [batch_size, input_size + hidden_size]

        Returns:
            A Tensor with a shape [batch_size, output_size]
        """
        return tf.matmul(input, w) + b

    def _while_op_lstm(self, input):
        shape = tf.shape(input)
        batch_size = shape[0]
        seq_len = shape[1]
        input_size = shape[2]

        self._i2h_shapes[0][0] = input_size
        w_gate = []
        b_gate = []
        with tf.name_scope('gate'):
            for shape in self._i2h_shapes:
                input_size = shape[0]
                hidden_size = shape[1]
                stddev = 1.0 / math.sqrt(hidden_size)

                # Concatenate input-to-hidden and hidden-to-hidden projections
                # into one matrix multiplication.
                w_gate.append(
                    tf.Variable(
                        tf.random.uniform(
                            [input_size + hidden_size, hidden_size],
                            minval=-stddev,
                            maxval=stddev)))

                b_gate.append(
                    tf.Variable(
                        tf.random.uniform(
                            [hidden_size], minval=-stddev, maxval=stddev)))

        w_output = []
        b_output = []
        with tf.name_scope('output'):
            for shape in self._i2h_shapes:
                input_size = shape[0]
                output_size = shape[1]
                stddev = 1.0 / math.sqrt(hidden_size)

                w_output.append(
                    tf.Variable(
                        tf.random.uniform(
                            [hidden_size, output_size],
                            minval=-stddev,
                            maxval=stddev)))
                b_output.append(
                    tf.Variable(
                        tf.random.uniform(
                            [output_size], minval=-stddev, maxval=stddev)))

        init_hidden = tf.zeros([batch_size, self._h2h_shapes[0][0]])
        init_cell = tf.zeros([batch_size, self._h2h_shapes[0][0]])

        init_t = tf.constant(0)
        init_output_array = tf.TensorArray(dtype=tf.float32, size=seq_len)
        cond = lambda i, _: tf.less(i, seq_len)

        def body(t, step):
            """The LSTM cell.
            """
            hidden, cell, output_array = step

            x_t = input[:, t]

            for depth in range(self.num_layers):
                combined = tf.concat([x_t, hidden], 1)

                f_gate = self._gate(combined, w_gate[depth], b_gate[depth])
                i_gate = self._gate(combined, w_gate[depth], b_gate[depth])
                o_gate = self._gate(combined, w_gate[depth], b_gate[depth])

                f_gate = tf.math.sigmoid(f_gate)
                i_gate = tf.math.sigmoid(i_gate)
                o_gate = tf.math.sigmoid(o_gate)

                cell = tf.math.add(
                    tf.math.multiply(cell, f_gate),
                    tf.math.multiply(
                        tf.math.tanh(
                            self._gate(combined, w_gate[depth],
                                       b_gate[depth])), i_gate))
                hidden = tf.math.multiply(tf.math.tanh(cell), o_gate)

                output_value = self._output(hidden, w_output[depth],
                                            b_output[depth])
                x_t = output_value

            return t + 1, (hidden, cell, output_array.write(t, x_t))

        _, step = tf.while_loop(
            cond, body, (init_t, (init_hidden, init_cell, init_output_array)))
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

        self.rnn = WhileOpLSTM([hidden_dim for _ in range(num_layers)],
                               hidden_dim)

    def call(self, input):
        x = self.emb(input)
        x = self.rnn(x)
        return self.linear(tf.reshape(x, self._output_shape))


def small_model(vocab_size):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=3)
