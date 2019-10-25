import math
import tensorflow as tf

layers = tf.keras.layers

__all__ = [
    'loss_fn',
    'Embedding',
    'FineGrainedOpLSTMCell',
]

import pdb


class FineGrainedOpLSTMCell(layers.Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super(FineGrainedOpLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def build(self, _):
        stddev = 1.0 / math.sqrt(self.hidden_size)
        with tf.name_scope('gate'):
            self.w_gates = [
                tf.Variable(
                    tf.random.uniform(
                        [self.input_size + self.hidden_size, self.hidden_size],
                        minval=-stddev,
                        maxval=stddev)) for _ in range(4)
            ]
            self.b_gates = [
                tf.Variable(
                    tf.random.uniform(
                        [self.hidden_size], minval=-stddev, maxval=stddev))
                for _ in range(4)
            ]

    def call(self, x, h_prev, c_prev):
        combined_x = tf.concat([x, h_prev], 1)
        f_gate = tf.matmul(combined_x, self.w_gates[0]) + self.b_gates[0]
        i_gate = tf.matmul(combined_x, self.w_gates[1]) + self.b_gates[1]
        o_gate = tf.matmul(combined_x, self.w_gates[2]) + self.b_gates[2]

        f_gate = tf.math.sigmoid(f_gate)
        i_gate = tf.math.sigmoid(i_gate)
        o_gate = tf.math.sigmoid(o_gate)

        cell_candidate = tf.math.tanh(
            tf.matmul(combined_x, self.w_gates[3]) + self.b_gates[3])
        cell = tf.math.add(
            tf.math.multiply(c_prev, f_gate),
            tf.math.multiply(cell_candidate, i_gate))
        hidden = tf.math.multiply(tf.math.tanh(cell), o_gate)
        return hidden, cell


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
