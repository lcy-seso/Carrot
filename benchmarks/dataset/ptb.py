from __future__ import print_function

import os
import codecs

import numpy as np
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
VOCAB_SUFFIX = "ptb.vocab.txt"
MODULE_NAME = "ptb"


def read_words(filename):
    """Read word strings from the given file.
    """
    with codecs.open(filepath, "r", encoding="utf-8") as f:
        return f.read().replace("\n", "<e>").split()


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
    with codecs.open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word_vocab[line.strip()] = i
    return word_vocab


def save_vocab(save_path, word_dict):
    with open(save_path, "w") as f:
        for k in word_dict:
            f.write("%s\n" % (k))


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


def file_to_word_ids(f, word_dict):
    UNK = word_dict['<unk>']
    return [
        word_dict.get(w, UNK) for w in f.read().replace('\n', '<e>').split()
    ]


def get_vocab(min_word_freq=None):
    prefix = "all" if min_word_freq is None else "cut%02d" % (min_word_freq)
    vocab_path = os.path.join(DATA_HOME, MODULE_NAME,
                              prefix + "_" + VOCAB_SUFFIX)
    if os.path.exists(vocab_path) and os.path.getsize(vocab_path):
        return load_vocab(vocab_path)
    else:
        word_dict = build_vocab(min_word_freq)
        save_vocab(vocab_path, word_dict)
        return word_dict


def lm_inputs(words, max_length, stride):
    cur_words = []
    next_words = []

    data_len = len(words)
    for i in range(0, data_len - max_length - 1, stride):
        cur_words.append(words[i:(i + max_length)])
        next_words.append(words[(i + 1):(i + max_length + 1)])
    return np.array(cur_words), np.array(next_words)


def train(max_length=50, stride=3, min_word_freq=None):
    with open(getfile(TRAIN_FILE), 'r') as f:
        return lm_inputs(
            file_to_word_ids(f, get_vocab(min_word_freq)), max_length, stride)


def test(max_length=50, stride=3, min_word_freq=None):
    with open(getfile(TEST_FILE), "r") as f:
        return lm_inputs(file_to_word_ids(f, get_vocab()), max_length, stride)


def valid(max_length=50, stride=3, min_word_freq=None):
    with open(getfile(VALID_FILE), "r") as f:
        return lm_inputs(file_to_word_ids(f, get_vocab()), max_length, stride)
