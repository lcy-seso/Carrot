from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import pdb

import numpy as np
import tensorflow as tf

layers = tf.keras.layers

__all__ = [
    'small_model',
]


def whileOpLSTM(input, hidden_size, output_size):
    """LSTM implemented in fine-grained operators via symbolic while-ops.
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


class Embedding(layers.Layer):
    """An Embedding layer."""

    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def build(self, _):
        self.embedding = self.add_variable(
            "embedding_kernel",
            shape=[self.vocab_size, self.embedding_dim],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True)

    def call(self, x):
        return tf.nn.embedding_lookup(self.embedding, x)


class PTBModel(tf.keras.Model):
    """LSTM for word language modeling.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 rnn_type,
                 seq_len=None,
                 batch_size=None):
        super(PTBModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.rnn_type = rnn_type
        self.embedding = Embedding(vocab_size, embedding_dim)

        if rnn_type == 'cudnn_lstm':
            self.rnn = StaticRNN(hidden_dim, num_layers, True)
        elif rnn_type == 'static_lstm':
            self.rnn = StaticRNN(hidden_dim, num_layers, False)
        elif rnn_type == 'while_op_lstm':
            assert tf.executing_eagerly() == False
            self.rnn = None
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


def loss_fn(model, inputs, targets):
    """Define the loss funtion.
    Args:
        inputs: Tensor(int64), the input tensor with a layout
            [batch_size, sequence_length].
        targets: Tensor(int64), the ground-truth with a layout
            [batch_size, sequence_length].
    Returns:
        The loss value which is scalar.
    """
    labels = tf.reshape(targets, [-1])
    outputs = model(inputs)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=outputs))


def small_model(vocab_size, rnn_type, batch_size=None, seq_len=None):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=3,
        rnn_type=rnn_type,
        batch_size=batch_size,
        seq_len=seq_len)
