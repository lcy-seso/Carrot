from __future__ import print_function

import os

import six as _six
import tarfile
import collections

from .common import download
from .common import DATA_HOME

__all__ = [
    'train',
    'test',
    'valid',
    'get_vocab',
]

PTB_URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
MD5 = '30177ea32e27c525793142b6bf2c8e2d'

TRAIN_FILE = "./simple-examples/data/ptb.train.txt"
TEST_FILE = "./simple-examples/data/ptb.test.txt"
VALID_FILE = "./simple-examples/data/ptb.valid.txt"
VOCAB_FILE = "ptb.vocab.txt"
MODULE_NAME = "ptb"


def getfile(tar_filename):
    # directly returns the saved local file instead of downloading it from
    # the Internet.
    local_path = os.path.join(DATA_HOME, MODULE_NAME,
                              os.path.split(tar_filename)[-1])
    if os.path.exists(local_path):
        return local_path

    # needed file does not exist, download it from the Internet.
    with tarfile.open(download(PTB_URL, MODULE_NAME, MD5), mode='r') as f:
        open(local_path, 'wb').write(f.extractfile(tar_filename).read())
    return local_path


def load_vocab(vocab_path):
    word_vocab = {}
    with open(vocab_path, "r") as f:
        for i, line in enumerate(f):
            word_vocab[line.strip()] = i
    return word_vocab


def word_count(f, word_freq=None):
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in f:
        for w in l.strip().split():
            word_freq[w] += 1
        word_freq['<s>'] += 1
        word_freq['<e>'] += 1

    return word_freq


def build_vocab(min_word_freq):
    """ Build the word vocabulary for PTB dataset.

    Build a word dictionary from the training data. Keys of the dictionary
    are words, and values are zero-based IDs of these words.
    """
    cut_freq = 0 if min_word_freq is None else min_word_freq
    with open(getfile(TRAIN_FILE)) as f:
        word_freq = word_count(f, word_count(f))
        if '<unk>' in word_freq:
            # remove <unk> for now, since we will set it as last index
            del word_freq['<unk>']

        word_freq = [x for x in _six.iteritems(word_freq) if x[1] > cut_freq]

        word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*word_freq_sorted))
        word_idx = dict(list(zip(words, _six.moves.range(len(words)))))
        word_idx['<unk>'] = len(words) + 1
    return word_idx


def data_reader(f, word_dict):
    UNK = word_dict['<unk>']
    dataset = []
    for i, line in enumerate(f):
        word_ids = [word_dict.get(w, UNK) for w in line.strip().split()]
        dataset.append(word_ids)
    return dataset


def get_vocab(min_word_freq=None):
    return build_vocab(min_word_freq)


def train():
    with open(getfile(TRAIN_FILE), "r") as f:
        return data_reader(f, get_vocab())


def test():
    with open(getfile(TEST_FILE), "r") as f:
        return data_reader(f, get_vocab())


def valid():
    with open(getfile(VALID_FILE), "r") as f:
        return data_reader(f, get_vocab())
