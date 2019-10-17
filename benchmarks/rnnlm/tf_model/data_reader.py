import collections

import tensorflow as tf

from dataset import ptb

__all__ = [
    'train_batch',
    'test_batch',
    'valid_batch',
    'vocab',
    'Input',
]


class Input(collections.namedtuple('Input', (
        'initializer',
        'x',
        'y',
))):
    pass


def vocab(min_word_freq=None):
    return ptb.get_vocab(min_word_freq)


def batch_input(x, y, batch_size, shuffle, eager_execution):
    x_dataset = tf.data.Dataset.from_tensor_slices(x)
    y_dataset = tf.data.Dataset.from_tensor_slices(y)

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100000 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if eager_execution:
        return dataset
    else:
        batched_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        cur_words, next_words = batched_iter.get_next()
        return Input(
            initializer=batched_iter.initializer, x=cur_words, y=next_words)


def train_batch(vocab,
                batch_size,
                max_length=50,
                stride=3,
                shuffle=False,
                eager_execution=True):
    x, y = ptb.train(vocab, max_length, stride)
    return batch_input(x, y, batch_size, shuffle, eager_execution)


def test_batch(vocab,
               batch_size,
               max_length=50,
               stride=3,
               shuffle=False,
               eager_execution=True):
    x, y = ptb.test(vocab, max_length, stride)
    return batch_input(x, y, batch_size, shuffle, eager_execution)


def valid_batch(vocab,
                batch_size,
                max_length=50,
                stride=3,
                shuffle=False,
                eager_execution=True):
    x, y = ptb.valid(vocab, max_length, stride)
    return batch_input(x, y, batch_size, shuffle, eager_execution)
