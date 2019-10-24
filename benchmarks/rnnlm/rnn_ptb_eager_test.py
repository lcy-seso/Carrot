import sys
import time
import gc

import tensorflow as tf

import tf_model.data_reader as reader
from tf_model.rnn_ptb import small_model
from tf_model import loss_fn
from test_utils import *


def force_gpu_sync():
    tf.constant(1).gpu().cpu()


class PTBBenchmarks(tf.test.Benchmark):
    BATCH_SIZE = 128
    SEQ_LEN = 50

    ITERS = 50

    def __init__(self):
        self.vocab = reader.vocab()

    def _report(self, test_name, start, num_iters, dev, batch_size):
        """
        Args:
            test_name (String): name of the test.
            start (String): timestamp of the start time.
            num_iters (Int): number of iterations.
            dev (String): device that the test is running. cpu or gpu.
            batch_size (Int): batch size.
        """
        total_time = time.time() - start
        wall_time = total_time / num_iters
        name = "%s_%s_batch_%d" % (test_name, dev, batch_size)
        examples_per_sec = batch_size / wall_time
        self.report_benchmark(
            iters=num_iters,
            wall_time=wall_time,
            name=name,
            extras={
                "examples_per_sec": examples_per_sec,
                "Time elapsed": total_time
            })

    def _benchmark_apply(self, dev, test_name, model):
        """Only Test the forward computation.
        """

        # TODO(Ying): use  synthetic data directly generated on the device
        # instead of using real data.
        inputs = reader.train_batch(
            self.vocab,
            PTBBenchmarks.BATCH_SIZE,
            max_length=PTBBenchmarks.SEQ_LEN,
            shuffle=False,
            eager_execution=True)

        with tf.device(device(dev)):
            for i, (x, y) in enumerate(inputs):  # Warmup
                if i == 10:
                    break
                model(x)
            gc.collect()

            start = time.time()
            for i, (x, y) in enumerate(inputs):
                if i == PTBBenchmarks.ITERS:
                    break
                model(x)
            self._report(test_name, start, PTBBenchmarks.ITERS, device(dev),
                         PTBBenchmarks.BATCH_SIZE)

    def benchmark_staticlstm_cpu_forward_small(self):
        self._benchmark_apply(
            "cpu", "eager_staticlstm_cpu_forward_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_staticlstm_gpu_forward_small(self):
        self._benchmark_apply(
            "gpu", "eager_staticlstm_cpu_forward_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_forward_small(self):
        self._benchmark_apply(
            "gpu", "eager_cudnn_lstm_forward_small",
            small_model(vocab_size=len(self.vocab), rnn_type="cudnn_lstm"))

    def _benchmark_train(self, dev, test_name, model):
        """Test both forward and backward.
        """

        def _step(x, y):
            with tf.GradientTape() as tape:
                loss_value = loss_fn(model, x, y)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        inputs = reader.train_batch(
            self.vocab,
            PTBBenchmarks.BATCH_SIZE,
            max_length=PTBBenchmarks.SEQ_LEN,
            shuffle=False,
            eager_execution=True)
        with tf.device(device(dev)):
            optimizer = tf.keras.optimizers.Adam(1.)

            for i, (x, y) in enumerate(inputs):  # Warmup
                if i == 10:
                    break
                _step(x, y)
            force_gpu_sync()
            gc.collect()

            start = time.time()
            for i, (x, y) in enumerate(inputs):
                if i == PTBBenchmarks.ITERS:
                    break
                _step(x, y)
            force_gpu_sync()
            self._report(test_name, start, PTBBenchmarks.ITERS, device(dev),
                         PTBBenchmarks.BATCH_SIZE)

    def benchmark_staticlstm_cpu_small(self):
        self._benchmark_train(
            "cpu", "eager_staticlstm_cpu_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_staticlstm_gpu_small(self):
        self._benchmark_train(
            "gpu", "eager_staticlstm_cpu_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_forward_small(self):
        self._benchmark_train(
            "gpu", "eager_cudnn_lstm_small",
            small_model(vocab_size=len(self.vocab), rnn_type="cudnn_lstm"))


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution(get_config())
    tf.test.main()
