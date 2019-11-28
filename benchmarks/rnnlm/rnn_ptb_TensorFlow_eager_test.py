import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only print error information.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import gc

import tensorflow as tf

import tf_model.data_reader as reader
import test_utils as tu
from tf_model.rnn_ptb import small_model
from tf_model import loss_fn


def force_gpu_sync():
    tf.constant(1).gpu().cpu()


class TFEagerPTBBenchmarks(tf.test.Benchmark):
    BATCH_SIZE = 128
    SEQ_LEN = 50
    ITERS = 50

    LOG_DEBUG_INFO = 0

    def __init__(self):
        tf.compat.v2.random.set_seed(1234)
        self.vocab = reader.vocab()
        self._init_logger()

    def _init_logger(self):
        self.logger = logging.getLogger("TF_eager_logger")
        self.logger.setLevel = (logging.DEBUG
                                if TFEagerPTBBenchmarks.LOG_DEBUG_INFO else
                                logging.INFO),
        self.logger.propagate = False
        fh = logging.FileHandler(
            filename="tensorflow_eager_ptb_rnn.log", mode="w")
        fh.setFormatter(logging.Formatter(fmt="%(message)s"))
        fh.setLevel(logging.DEBUG
                    if TFEagerPTBBenchmarks.LOG_DEBUG_INFO else logging.INFO)
        self.logger.addHandler(fh)

    def _report(self, test_name, start):
        """
        Args:
            test_name (String): Name of the test.
            start (String): Timestamp of the start time.
        """
        elapsed_time = time.time() - start
        average_time = elapsed_time / TFEagerPTBBenchmarks.ITERS
        seq_per_sec = (TFEagerPTBBenchmarks.ITERS *
                       TFEagerPTBBenchmarks.BATCH_SIZE) / elapsed_time
        self.logger.info(("|%s|%.4f\t|%.4f\t|%.4f|") %
                         (test_name, average_time, elapsed_time, seq_per_sec))

    def _benchmark_apply(self, dev, test_name, model):
        """Only Test the forward computation.
        Args:
            dev, String: Device that on which the test is running. cpu or gpu.
            test_name, String: Name of the test.
            model, Callable: The tested model. It should be a callable object.
        """

        # TODO(Ying): use  synthetic data directly generated on the device
        # instead of using real data.
        inputs = reader.train_batch(
            self.vocab,
            TFEagerPTBBenchmarks.BATCH_SIZE,
            max_length=TFEagerPTBBenchmarks.SEQ_LEN,
            shuffle=False,
            eager_execution=True)

        with tf.device(tu.device(dev)):
            for i, (x, y) in enumerate(inputs):  # Warmup
                if i == 10:
                    break
                model(x)
            gc.collect()

            start = time.time()
            for i, (x, y) in enumerate(inputs):
                if i == TFEagerPTBBenchmarks.ITERS:
                    break
                model(x)
            self._report(test_name, start)

    def benchmark_fine_grained_op_lstm_forward(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_apply(
                device, f"eager_finegrained_op_lstm_{device}_forward",
                small_model(
                    vocab_size=len(self.vocab),
                    rnn_type="fine_grained_op_lstm"))

    def benchmark_staticlstm_cpu_forward_small(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_apply(
                device, f"eager_staticlstm_{device}_forward",
                small_model(
                    vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_forward_small(self):
        self._benchmark_apply(
            "gpu", "eager_cudnnlstm_forward_small",
            small_model(vocab_size=len(self.vocab), rnn_type="cudnn_lstm"))

    def _benchmark_train(self, dev, test_name, model):
        """Test both forward and backward.
        Args:
            dev, String: Device that on which the test is running. cpu or gpu.
            test_name, String: Name of the test.
            model, Callable: The tested model. It should be a callable object.
        """

        def _step(x, y):
            with tf.GradientTape() as tape:
                loss_value = loss_fn(model, x, y)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        inputs = reader.train_batch(
            self.vocab,
            TFEagerPTBBenchmarks.BATCH_SIZE,
            max_length=TFEagerPTBBenchmarks.SEQ_LEN,
            shuffle=False,
            eager_execution=True)
        with tf.device(tu.device(dev)):
            optimizer = tf.keras.optimizers.Adam(1.)

            for i, (x, y) in enumerate(inputs):  # Warmup
                if i == 10:
                    break
                _step(x, y)
            if dev == "gpu":
                force_gpu_sync()
            gc.collect()

            start = time.time()
            for i, (x, y) in enumerate(inputs):
                if i == TFEagerPTBBenchmarks.ITERS:
                    break
                _step(x, y)
            force_gpu_sync()
            self._report(test_name, start)

    def benchmark_fine_grained_op_lstm_train(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_train(
                device, f"eager_finegrained_op_lstm_{device}_train",
                small_model(
                    vocab_size=len(self.vocab),
                    rnn_type="fine_grained_op_lstm"))

    def benchmark_staticlstm_train(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_train(
                device, f"eager_staticlstm_{device}_train",
                small_model(
                    vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_train(self):
        self._benchmark_train(
            "gpu", "eager_cudnnlstm_train",
            small_model(vocab_size=len(self.vocab), rnn_type="cudnn_lstm"))


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution(tu.get_config())
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.test.main()
