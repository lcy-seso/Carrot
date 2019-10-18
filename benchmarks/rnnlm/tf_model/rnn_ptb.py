from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

layers = tf.keras.layers

__all__ = [
    'small_model',
    'large_model',
    'test_model',
]


class RNN(tf.keras.Model):
    """A static RNN.
    """

    def __init__(self, hidden_dim, num_layers, batch_size=None):
        super(RNN, self).__init__()

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
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.0,
                recurrent_dropout=0.0,
                implementation=2) for _ in range(num_layers)
        ]
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

    def call(self, input_seq):
        """Define computations in a single time step.

        input_seq: Tensor, the layout is
            [batch_size, max_sequence_length, embedding_dim].
        """

        # NOTE: below line only works in eager mode.
        # In eager mode, TensorFlow operations are immediately evaluated and
        # return their values to Python, so below line will return the vaule
        # of batch size, while in graph mode, we use dataset API to feed data,
        # the batch size dimension is None when defining the graph.
        batch_size = int(input_seq.shape[0])

        # A workaround to make the model definion work in both eager and
        # graph mode.
        batch_size = self.batch_size if batch_size is None else batch_size
        assert batch_size is not None

        for c in self.cells:
            state = (tf.zeros((batch_size, self.hidden_dim)),
                     tf.zeros((batch_size, self.hidden_dim)))
            outputs = []

            # unpack the input 3D tensors along the `max_sequence_length` axis
            # to get input tensors for each time step.
            input_seq = tf.unstack(
                input_seq, num=int(input_seq.shape[1]), axis=1)
            for inp in input_seq:
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
                 batch_size=None,
                 use_cudnn_rnn=False):
        super(PTBModel, self).__init__()

        self.use_cudnn_rnn = use_cudnn_rnn
        self.embedding = Embedding(vocab_size, embedding_dim)

        if self.use_cudnn_rnn:
            self.rnn = CudnnLSTM(num_layers, hidden_dim, dropout=0.)
        else:
            self.rnn = RNN(hidden_dim, num_layers, batch_size)

        self.linear = layers.Dense(
            vocab_size,
            kernel_initializer=tf.keras.initializers.glorot_normal())
        self._output_shape = [-1, embedding_dim]

    def call(self, input_seq):
        """Run the forward pass of PTBModel.
        Args:
            input_seq: Tensor(int64), layout: [batch_size, sequence_length].
        Returns:
            outputs tensors of inference.
        """
        y = self.embedding(input_seq)
        y = self.rnn(y)[0]
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


def small_model(vocab_size, use_cudnn_rnn):
    """Returns a PTBModel with a small configuration."""
    return PTBModel(
        vocab_size=vocab_size,
        embedding_dim=200,
        hidden_dim=200,
        num_layers=3,
        use_cudnn_rnn=use_cudnn_rnn)


def large_model(vocab_size, use_cudnn_rnn):
    """Returns a PTBModel with a large configuration."""
    return PTBModel(
        vocab_size=vocab_size,
        embedding_dim=650,
        hidden_dim=650,
        num_layers=3,
        use_cudnn_rnn=use_cudnn_rnn)


def test_model(vocab_size, embedding_dim, hidden_dim, num_layers,
               use_cudnn_rnn):
    """Returns a PTBModel with configurable configuration."""
    return PTBModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_cudnn_rnn=use_cudnn_rnn)
