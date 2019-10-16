import gc
import sys

import click

import tensorflow as tf

from tf_model import data_reader
from tf_model.rnn_ptb import PTBModel, train


def device():
    return "/device:GPU:0" if tf.test.is_gpu_available(
        cuda_only=True) else "/device:CPU:0"


@click.command("train rnn lm.")
@click.option("--logdir", default="", help="Directory for checkpoint.")
@click.option("--epoch", default=10, help="Number of epochs.")
@click.option("--learning_rate", default=0.1, help="The learning rate.")
@click.option(
    "--batch_size",
    default=20,
    help="The number of training examples in one forward/backward pass.")
@click.option("--max_seq_len", default=50, help="Sequence length.")
@click.option("--embedding_dim", default=200, help="Embedding dimension.")
@click.option("--hidden_dim", default=200, help="Hidden layer dimension.")
@click.option("--num_layers", default=3, help="Number of RNN layers.")
@click.option("--use_gpu", default=False, help="Use GPU or not.")
@click.option(
    "--use_cudnn_rnn",
    default=False,
    help="Disable the fast CuDNN RNN (when no gpu)")
def main(logdir, epoch, learning_rate, batch_size, max_seq_len, embedding_dim,
         hidden_dim, num_layers, use_cudnn_rnn, use_gpu):
    vocab = data_reader.vocab()
    train_data = data_reader.train_batch(
        vocab, batch_size, max_length=max_seq_len, shuffle=False)

    model = PTBModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_cudnn_rnn=use_cudnn_rnn)

    with tf.device(device()):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        train(model, optimizer, train_data, epoch)


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
