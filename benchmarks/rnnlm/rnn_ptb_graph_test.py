import time
import sys

import tensorflow as tf

import tf_model.data_reader as reader
from tf_model.rnn_ptb import small_model, loss_fn
from test_utils import *


class PTBGraphModeTest(tf.test.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.max_seq_len = 50
        self.learning_rate = 1e-2

        self.vocab = reader.vocab()

        self.WARMUP = 10
        self.TEST_BATCHES = 30

    def testTrain(self):
        with tf.Graph().as_default():
            with tf.device("/device:CPU:0"):
                # dataset API only works on CPU.
                inputs = reader.train_batch(
                    self.vocab,
                    self.batch_size,
                    max_length=self.max_seq_len,
                    shuffle=False,
                    eager_execution=False)

            with tf.device(device()):
                model = small_model(
                    vocab_size=len(self.vocab), use_cudnn_rnn=False)
                loss = loss_fn(model, inputs.x, inputs.y)
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate)
                train_op = optimizer.minimize(loss)

                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    sess.run(inputs.initializer)

                    batch_id = 1
                    while True:
                        try:
                            loss_value, _ = sess.run([loss, train_op])
                            print("Batch %d, loss = %.4f" % (batch_id,
                                                             loss_value))

                            if batch_id >= self.TEST_BATCHES:
                                print("Time elapsed: %.4f" %
                                      (time.time() - self.start))
                                break

                            if batch_id == self.WARMUP:
                                self.start = time.time()

                            batch_id = batch_id + 1

                        except tf.errors.OutOfRangeError:
                            sess.run(inputs.initializer)
                            continue


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
