from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

layers = tf.keras.layers

__all__ = [
    'PTBModel',
]


class RNN(tf.keras.Model):
    """A static RNN.
    """

    def __init__(self, hidden_dim, num_layers):
        super(RNN, self).__init__()
        self.cells = [
            tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim)
            for _ in range(num_layers)
        ]

    def call(self, input_seq):
        batch_size = int(input_seq.shape[1])
        for c in self.cells:
            state = c.zero_state(batch_size, tf.float32)
            outputs = []
            input_seq = tf.unstack(
                input_seq, num=int(input_seq.shape[0]), axis=0)
            for inp in input_seq:
                output, state = c(inp, state)
                outputs.append(output)

            input_seq = tf.stack(outputs, axis=0)
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
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
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
                 use_cudnn_rnn=False):
        super(PTBModel, self).__init__()

        self.use_cudnn_rnn = use_cudnn_rnn
        self.embedding = Embedding(vocab_size, embedding_dim)

        if self.use_cudnn_rnn:
            self.rnn = CudnnLSTM(num_layers, hidden_dim, dropout=0.)
        else:
            self.rnn = RNN(hidden_dim, num_layers)

        self.linear = layers.Dense(
            vocab_size,
            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._output_shape = [-1, embedding_dim]

    def call(self, input_seq):
        """Run the forward pass of PTBModel.
        Args:
            input_seq: [length, batch] shape int64 tensor.
        Returns:
            outputs tensors of inference.
        """
        y = self.embedding(input_seq)
        y = self.rnn(y)[0]
        return self.linear(tf.reshape(y, self._output_shape))


def loss_fn(model, inputs, targets):
    labels = tf.reshape(targets, [-1])
    outputs = model(inputs)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=outputs))


def train(model, optimizer, train_data, epoch):
    """training an epoch."""
    for i in range(epoch):
        for (batch, (x, y)) in enumerate(train_data):
            # train step.
            with tf.GradientTape() as tape:
                loss_value = loss_fn(model, x, y)
                print("Epoch %d, batch %02d, loss = %.4f" % (i, batch,
                                                             loss_value))

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
