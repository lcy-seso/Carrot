import tensorflow as tf

layers = tf.keras.layers

__all__ = [
    'loss_fn',
]


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
