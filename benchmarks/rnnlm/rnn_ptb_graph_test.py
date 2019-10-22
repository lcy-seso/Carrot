import time
import sys

import tensorflow as tf

import tf_model.data_reader as reader
from tf_model.rnn_ptb import small_model, loss_fn
from test_utils import *


def train(inputs, loss, train_op, n_step):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(inputs.initializer)

        batch_id = 1
        while True:
            try:
                loss_value, _ = sess.run([loss, train_op])
                print("Batch %d, loss = %.4f" % (batch_id, loss_value))

                if batch_id == n_step:
                    break

                batch_id = batch_id + 1
            except tf.errors.OutOfRangeError:
                sess.run(inputs.initializer)
                continue


class PTBGraphModeTest(tf.test.TestCase):
    def setUp(self):
        self.batch_size = 128
        self.max_seq_len = 50
        self.learning_rate = 1e-2

        self.vocab = reader.vocab()

        self.WARMUP = 10
        self.TEST_BATCHES = 20

    def testWhileOpLSTMCPU(self):
        with tf.Graph().as_default():
            with tf.device("/device:CPU:0"):
                inputs = reader.train_batch(
                    self.vocab,
                    self.batch_size,
                    max_length=self.max_seq_len,
                    shuffle=False,
                    eager_execution=False)

                model = small_model(vocab_size=len(self.vocab), rnn_type='')
                loss = loss_fn(model, inputs.x, inputs.y)
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate)
                train_op = optimizer.minimize(loss)

                train(inputs, loss, train_op, self.WARMUP)
                self.start = time.time()
                train(inputs, loss, train_op, self.TEST_BATCHES)
                print("Time elapsed: %.4f" % (time.time() - self.start))

    # def testGraphCPUStaticLSTM(self):
    #     with tf.Graph().as_default():
    #         with tf.device("/device:CPU:0"):
    #             inputs = reader.train_batch(
    #                 self.vocab,
    #                 self.batch_size,
    #                 max_length=self.max_seq_len,
    #                 shuffle=False,
    #                 eager_execution=False)

    #             model = small_model(
    #                 vocab_size=len(self.vocab), rnn_type='static_lstm')
    #             loss = loss_fn(model, inputs.x, inputs.y)
    #             optimizer = tf.compat.v1.train.AdamOptimizer(
    #                 self.learning_rate)
    #             train_op = optimizer.minimize(loss)

    #             train(inputs, loss, train_op, self.WARMUP)
    #             self.start = time.time()
    #             train(inputs, loss, train_op, self.TEST_BATCHES)
    #             print("Time elapsed: %.4f" % (time.time() - self.start))

    # def testGraphGPUStaticLSTM(self):
    #     with tf.Graph().as_default():
    #         with tf.device("/device:CPU:0"):
    #             inputs = reader.train_batch(
    #                 self.vocab,
    #                 self.batch_size,
    #                 max_length=self.max_seq_len,
    #                 shuffle=False,
    #                 eager_execution=False)

    #         with tf.device("/device:GPU:0"):
    #             model = small_model(
    #                 vocab_size=len(self.vocab), rnn_type='static_lstm')
    #             loss = loss_fn(model, inputs.x, inputs.y)
    #             optimizer = tf.compat.v1.train.AdamOptimizer(
    #                 self.learning_rate)
    #             train_op = optimizer.minimize(loss)

    #             # warm up batches.
    #             train(inputs, loss, train_op, self.WARMUP)

    #             # test batches
    #             force_gpu_sync()
    #             self.start = time.time()
    #             train(inputs, loss, train_op, self.TEST_BATCHES)
    #             print("Time elapsed: %.4f" % (time.time() - self.start))

    # def testGraphCuDNNLSTM(self):
    #     with tf.Graph().as_default():
    #         with tf.device("/device:CPU:0"):
    #             inputs = reader.train_batch(
    #                 self.vocab,
    #                 self.batch_size,
    #                 max_length=self.max_seq_len,
    #                 shuffle=False,
    #                 eager_execution=False)

    #         with tf.device("/device:GPU:0"):
    #             model = small_model(
    #                 vocab_size=len(self.vocab), rnn_type='cudnn_lstm')
    #             loss = loss_fn(model, inputs.x, inputs.y)
    #             optimizer = tf.compat.v1.train.AdamOptimizer(
    #                 self.learning_rate)
    #             train_op = optimizer.minimize(loss)

    #             train(inputs, loss, train_op, self.WARMUP)
    #             self.start = time.time()
    #             train(inputs, loss, train_op, self.TEST_BATCHES)
    #             print("Time elapsed: %.4f" % (time.time() - self.start))


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
