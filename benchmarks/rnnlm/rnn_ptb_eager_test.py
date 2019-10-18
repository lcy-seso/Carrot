import sys
import time

import tensorflow as tf

import tf_model.data_reader as reader
from tf_model.rnn_ptb import small_model, loss_fn
from test_utils import *


class PTBEagerTest(tf.test.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.max_seq_len = 50
        self.learning_rate = 1e-2

        self.vocab = reader.vocab()
        self.WARMUP = 10
        self.TEST_BATCHES = 30

        self.start = None

    def testTrain(self):
        train_data = reader.train_batch(
            self.vocab,
            self.batch_size,
            max_length=self.max_seq_len,
            shuffle=False)
        with tf.device(device()):
            model = small_model(
                vocab_size=len(self.vocab), use_cudnn_rnn=False)
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)

            for (batch, (x, y)) in enumerate(train_data):
                with tf.GradientTape() as tape:
                    loss_value = loss_fn(model, x, y)
                    print("Batch %d, loss = %.4f" % (batch + 1, loss_value))

                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                if batch == self.WARMUP - 1:
                    self.start = time.time()

                if batch == self.TEST_BATCHES - 1:
                    print("Time elapsed: %.4f" % (time.time() - self.start))
                    break


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution(get_config())
    tf.test.main()
