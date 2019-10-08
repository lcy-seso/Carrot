from math import sqrt

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from Utils.data.ptb import vocab

ntokens = len(vocab.itos)
output_size = ntokens

# shape of `model_data` is (seq_len, batch_size)
model_data = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
# shape of `model_target` is (seq_len, batch_size)
model_target = tf.compat.v1.placeholder(tf.int64, shape=[None, None])


def LSTM(input, input_size: int, hidden_size: int, output_size: int):
    """
    Args:
        input(Tensor): the shape is (seq_len, batch_size, input_size)
        input_size(int): input dimension
        hidden_size(int): hidden/cell dimension
        output_size(int): output dimension
    """
    shape = tf.shape(input)
    seq_len = shape[0]
    batch_size = shape[1]

    stddev = 1.0 / sqrt(hidden_size)

    with tf.name_scope('gate'):
        # shape of `w_gate` is (input_size + hidden_size, hidden_size)
        w_gate = tf.Variable(
            tf.random.uniform(
                [input_size + hidden_size, hidden_size],
                minval=-stddev,
                maxval=stddev))

        # shape of `b_gate` is (hidden_size)
        b_gate = tf.Variable(
            tf.random.uniform([hidden_size], minval=-stddev, maxval=stddev))

    with tf.name_scope('output'):
        # shape of `w_output` is (hidden_size, output_size)
        w_output = tf.Variable(
            tf.random.uniform(
                [hidden_size, output_size], minval=-stddev, maxval=stddev))

        # shape of `b_output` is (output_size)
        b_output = tf.Variable(
            tf.random.uniform([output_size], minval=-stddev, maxval=stddev))

    def gate(input):
        # shape of `input` is (batch_size, input_size + hidden_size)
        # shape of the return value is (batch_size, hidden_size)
        return tf.matmul(input, w_gate) + b_gate

    def output(input):
        # shape of `input` is (batch_size, input_size + hidden_size)
        # shape of the return value is (batch_size, output_size)
        return tf.matmul(input, w_output) + b_output

    # Start constructing while loop
    # shape of `init_hidden` is (batch_size, hidden_size)
    init_hidden = tf.zeros([batch_size, hidden_size])
    init_cell = tf.zeros([batch_size, hidden_size])
    init_i = tf.constant(0)
    init_output_array = tf.TensorArray(dtype=tf.float32, size=seq_len)

    cond = lambda i, _: tf.less(i, seq_len)

    def body(i: int, step):
        # step: (hidden, cell, output_array)

        # shape of `hidden` is (batch_size, hidden_size)
        # shape of `cell` is (batch_size, hidden_size)
        hidden, cell, output_array = step

        # shape of `x_t` is (batch_size, input_size)
        x_t = input[i]

        # shape of `combined` is (batch_size, input_size + hidden_size)
        combined = tf.concat([x_t, hidden], 1)

        # shape of `f_gate`/`i_gate`/`o_gate` is (batch_size, hidden_size)
        f_gate = gate(combined)
        i_gate = gate(combined)
        o_gate = gate(combined)

        f_gate = tf.math.sigmoid(f_gate)
        i_gate = tf.math.sigmoid(i_gate)
        o_gate = tf.math.sigmoid(o_gate)

        # shape of `cell` is (batch_size, hidden_size)
        cell = tf.math.add(
            tf.math.multiply(cell, f_gate),
            tf.math.multiply(tf.math.tanh(gate(combined)), i_gate))

        # shape of `hidden` is (batch_size, hidden_size)
        hidden = tf.math.multiply(tf.math.tanh(cell), o_gate)

        # shape of `output` is (batch_size, output_size)
        output_value = output(hidden)

        return i + 1, (hidden, cell, output_array.write(i, output_value))

    _, step = tf.while_loop(
        cond, body, (init_i, (init_hidden, init_cell, init_output_array)))
    _, _, output_array = step
    # Finish constructing while loop

    # shape of the return value is (seq_len, batch_size, output_size)
    return output_array.stack()


def get_model(embedding_size: int, hidden_size: int, lr: int):
    word_embedding = tf.compat.v1.get_variable("word_embedding",
                                               [ntokens, embedding_size])
    # shape of `emb` is (batch_size, embedding_size)
    emb = tf.nn.embedding_lookup(word_embedding, model_data)

    # shape of output is (seq_len, batch_size, output_size)
    output = LSTM(emb, embedding_size, hidden_size, output_size)

    with tf.name_scope('Cross_Entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=model_target))

    with tf.compat.v1.name_scope('Train'):
        train_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(
            cross_entropy)

    return train_step, model_data, model_target, cross_entropy
