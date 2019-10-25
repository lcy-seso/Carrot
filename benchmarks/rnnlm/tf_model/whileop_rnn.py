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


class WhileOpLSTMNet(tf.keras.Model):
    """LSTM implemented in fine-grained operators via symbolic while-ops.

    Only works in graph-mode.
    """

    def __init__(self, hiddens, output_size):
        """
        Args:
            input, Tensor: [batch_size, seq_len, input_dim]
            hiddens, List of int: hidden and also the cell dimension for each
                stacked RNN layer.
            output_size, int: output dimension.
        Return:
            A Tensor with a shape [batch_size, sequence_length, hidden_dim]
        """
        super(WhileOpLSTMNet, self).__init__()

        self.hiddens = hiddens
        self._i2h_shapes = [[h, h] for h in hiddens]
        self._h2h_shapes = [[h, h] for h in hiddens]
        self._h2o_shapes = [[h, h] for h in hiddens]
        self._h2o_shapes[-1][-1] = output_size
        self.num_layers = len(hiddens)

    def _while_op_lstm(self, input):
        shape = tf.shape(input)
        batch_size = shape[0]
        seq_len = shape[1]
        input_size = shape[2]

        self._i2h_shapes[0][0] = input_size
        w_gate = []
        b_gate = []
        with tf.name_scope('gate'):
            for shape in self._i2h_shapes:  # depth
                input_size = shape[0]
                hidden_size = shape[1]
                stddev = 1.0 / math.sqrt(hidden_size)

                # Concatenate input-to-hidden and hidden-to-hidden
                # projections into one matrix multiplication.
                w_gate.append([
                    tf.Variable(
                        tf.random.uniform(
                            [input_size + hidden_size, hidden_size],
                            minval=-stddev,
                            maxval=stddev)) for _ in range(4)
                ])

                b_gate.append([
                    tf.Variable(
                        tf.random.uniform(
                            [hidden_size], minval=-stddev, maxval=stddev))
                    for _ in range(4)
                ])

        w_output = []
        b_output = []
        with tf.name_scope('output'):
            for shape in self._i2h_shapes:
                input_size = shape[0]
                output_size = shape[1]
                stddev = 1.0 / math.sqrt(hidden_size)

                w_output.append([
                    tf.Variable(
                        tf.random.uniform(
                            [hidden_size, output_size],
                            minval=-stddev,
                            maxval=stddev)) for _ in range(4)
                ])
                b_output.append([
                    tf.Variable(
                        tf.random.uniform(
                            [output_size], minval=-stddev, maxval=stddev))
                    for _ in range(4)
                ])

        init_hidden = [tf.zeros([batch_size, self._h2h_shapes[0][0]])
                       ] * self.num_layers
        init_cell = [tf.zeros([batch_size, self._h2h_shapes[0][0]])
                     ] * self.num_layers

        init_t = tf.constant(0)
        init_output_array = tf.TensorArray(dtype=tf.float32, size=seq_len)
        cond = lambda i, _: tf.less(i, seq_len)

        def body(t, step):
            """The LSTM cell.
            For some TF implementation constrains, we cannot reuse LSTMCell
            defined in utils.py, but implement in the body function.
            """
            hiddens_prev, cells_prev, output_array = step

            x_t = input[:, t]

            cells = []
            hiddens = []
            for depth in range(len(hiddens_prev)):
                combined_x = tf.concat([x_t, hiddens_prev[depth]], 1)

                f_gate = tf.matmul(combined_x,
                                   w_gate[depth][0]) + b_gate[depth][0]
                i_gate = tf.matmul(combined_x,
                                   w_gate[depth][1]) + b_gate[depth][1]
                o_gate = tf.matmul(combined_x,
                                   w_gate[depth][2]) + b_gate[depth][2]

                f_gate = tf.math.sigmoid(f_gate)
                i_gate = tf.math.sigmoid(i_gate)
                o_gate = tf.math.sigmoid(o_gate)

                cell_candidate = tf.math.tanh(
                    tf.matmul(combined_x, w_gate[depth][3]) + b_gate[depth][3])
                cell = tf.math.add(
                    tf.math.multiply(cells_prev[depth], f_gate),
                    tf.math.multiply(cell_candidate, i_gate))
                hidden = tf.math.multiply(tf.math.tanh(cell), o_gate)

                hiddens.append(hidden)
                cells.append(cell)
                x_t = hidden

            return t + 1, (hiddens, cells, output_array.write(t, x_t))

        _, step = tf.while_loop(
            cond, body, (init_t, (init_hidden, init_cell, init_output_array)))
        _, _, output_array = step

        return output_array.stack()

    def __call__(self, input_seq):
        """Stacked LSTM network implemented by TF's symbolic while loop operator.
        Args:
            input_seq, Tensor, input sequence batch. The layout must be
                batch_size major: [batch_size, seq_len, input_dim].
        """
        return self._while_op_lstm(input_seq)


class PTBModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(PTBModel, self).__init__()

        self.emb = Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size

        self.linear = layers.Dense(
            vocab_size,
            kernel_initializer=tf.keras.initializers.glorot_normal())
        self.rnn = WhileOpLSTMNet([hidden_size] * num_layers, hidden_size)
        self._output_shape = [-1, hidden_size]

    def call(self, input):
        x = self.emb(input)
        x = self.rnn(x)
        return self.linear(tf.reshape(x, self._output_shape))


def small_model(vocab_size):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=3)
