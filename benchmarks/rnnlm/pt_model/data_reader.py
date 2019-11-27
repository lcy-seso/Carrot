from numpy import ndarray

import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import ptb

__all__ = [
    'train_loader',
    'vocab',
]


def vocab(min_word_freq=None):
    return ptb.get_vocab(min_word_freq)


def batch_input(x: ndarray, y: ndarray, batch_size: int, shuffle: bool,
                device: str):
    dataset = TensorDataset(
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def train_loader(vocab,
                 batch_size,
                 max_length=50,
                 stride=3,
                 shuffle=False,
                 device="cpu"):
    x, y = ptb.train(vocab, max_length, stride)
    return batch_input(x, y, batch_size, shuffle, device)
