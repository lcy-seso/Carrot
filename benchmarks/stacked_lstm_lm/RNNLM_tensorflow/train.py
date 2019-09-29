import argparse
import sys

sys.path.append('../../Utils/Python')

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from torchtext.data import Iterator

from model import get_model
from Utils.data.ptb import train_dataset, valid_dataset, test_dataset, vocab


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='upper epoch limit')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=200,
        help='size of word embeddings')
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=200,
        help='number of hidden units per layer')

    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        metavar='N',
        help='report interval')

    args = parser.parse_args()

    train_loader, valid_loader, test_loader = Iterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        shuffle=True,
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.text))

    train_step, model_data, model_target, cross_entropy = get_model(
        args.embedding_size, args.hidden_size, args.lr)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(args.epochs):
            print("Batch", i)
            for batch_idx, batch in enumerate(train_loader):
                data, target = batch.text.numpy(), batch.target.numpy()
                _ = sess.run(
                    train_step,
                    feed_dict={
                        model_data: data,
                        model_target: target
                    })

                if batch_idx % args.log_interval == 0:
                    loss = sess.run(
                        cross_entropy,
                        feed_dict={
                            model_data: data,
                            model_target: target
                        })
                    print("Iter", batch_idx, ", Minibatch Loss=", loss)

            batch = next(test_loader.__iter__())
            test_data, test_target = batch.text.numpy(), batch.target.numpy()
            loss = sess.run(
                cross_entropy,
                feed_dict={
                    model_data: test_data,
                    model_target: test_target
                })
            print("Test Loss", loss)


if __name__ == '__main__':
    main()
