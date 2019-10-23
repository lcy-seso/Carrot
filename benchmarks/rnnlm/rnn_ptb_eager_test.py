import sys
import time
import gc

import tensorflow as tf

import tf_model.data_reader as reader
from tf_model.rnn_ptb import small_model, loss_fn
from test_utils import get_config


def force_gpu_sync():
    tf.constant(1).gpu().cpu()


def train(model, optimizer, train_data, n_step):
    for (batch, (x, y)) in enumerate(train_data):
        with tf.GradientTape() as tape:
            loss_value = loss_fn(model, x, y)
            print("Batch %d, loss = %.4f" % (batch + 1, loss_value))

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if batch == n_step - 1:
            break


class PTBEagerModeTest(tf.test.TestCase):
    def setUp(self):
        self.batch_size = 128
        self.max_seq_len = 50
        self.learning_rate = 1e-2

        self.vocab = reader.vocab()
        self.WARMUP = 10
        self.TEST_BATCHES = 20

        self.start = None

    def testEagerCPUStaticLSTM(self):
        with tf.device('/device:CPU:0'):
            train_data = reader.train_batch(
                self.vocab,
                self.batch_size,
                max_length=self.max_seq_len,
                shuffle=False)

            model = small_model(
                vocab_size=len(self.vocab), rnn_type='static_lstm')
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)

            # warm step.
            train(model, optimizer, train_data, self.WARMUP)

            self.start = time.time()

            # time batches.
            train(model, optimizer, train_data, self.TEST_BATCHES)
            print("Time elapsed: %.4f" % (time.time() - self.start))

    def testEagerGPUStaticLSTM(self):
        train_data = reader.train_batch(
            self.vocab,
            self.batch_size,
            max_length=self.max_seq_len,
            shuffle=False)
        with tf.device('/device:GPU:0'):
            model = small_model(
                vocab_size=len(self.vocab), rnn_type='static_lstm')
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)

            # warm step.
            train(model, optimizer, train_data, self.WARMUP)

            force_gpu_sync()
            self.start = time.time()

            # time batches.
            train(model, optimizer, train_data, self.TEST_BATCHES)
            print("Time elapsed: %.4f" % (time.time() - self.start))

    def testEagerCudnnLSTM(self):
        train_data = reader.train_batch(
            self.vocab,
            self.batch_size,
            max_length=self.max_seq_len,
            shuffle=False)
        with tf.device('device:GPU:0'):
            model = small_model(
                vocab_size=len(self.vocab), rnn_type='cudnn_lstm')
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)

            # warm step.
            train(model, optimizer, train_data, self.WARMUP)

            force_gpu_sync()
            self.start = time.time()

            # time batches.
            train(model, optimizer, train_data, self.TEST_BATCHES)
            print("Time elapsed: %.4f" % (time.time() - self.start))


if __name__ == "__main__":
    gc.disable()
    tf.compat.v1.enable_eager_execution(get_config())
    tf.test.main()
