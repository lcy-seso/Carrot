import tensorflow as tf

from dataset import ptb

__all__ = [
    'train_batch',
    'test_batch',
    'valid_batch',
]


def vocab(min_word_freq=None):
    return ptb.get_vocab(min_word_freq)


def batch_input(dataset, batch_size, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000 * batch_size)
    return dataset.batch(batch_size)


def train_batch(vocab, batch_size, max_length=50, stride=3, shuffle=False):
    return batch_input(
        ptb.train(vocab, max_length, stride), batch_size, shuffle)


def test_batch(vocab, batch_size, max_length=50, stride=3, shuffle=False):
    return batch_input(
        ptb.test(vocab, max_length, stride), batch_size, shuffle)


def valid_batch(vocab, batch_size, max_length=50, stride=3, shuffle=False):
    return batch_input(
        ptb.valid(vocab, max_length, stride), batch_size, shuffle)
