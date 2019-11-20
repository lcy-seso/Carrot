import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only print error information.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import sys
import time

import tensorflow as tf

import tf_model.data_reader as reader
import tf_test_utils as tu
from tf_model import loss_fn
from tf_model.rnn_ptb import small_model
import tf_model.whileop_rnn as m


class PTBBenchmarks(tf.test.Benchmark):
    BATCH_SIZE = 128
    SEQ_LEN = 50
    ITERS = 50

    LOG_DEBUG = True

    def __init__(self):
        tf.compat.v2.random.set_seed(1234)
        self.vocab = reader.vocab()
        self._init_logger()

    def _init_logger(self):
        self.logger = logging.getLogger("TF_graph_logger")
        fh = logging.FileHandler("tensorflow_graph_ptb_rnn.log", mode="w")
        fh.setLevel(logging.DEBUG if PTBBenchmarks.LOG_DEBUG else logging.INFO)
        fh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fh)
        self.logger = logging.getLogger("TF_graph_logger")

    def _report(self, test_name, start):
        """
        Args:
            test_name (String): Name of the test.
            start (String): Timestamp of the start time.
            num_iters (Int): Number of tested iterations.
        """
        elapsed_time = time.time() - start
        average_time = elapsed_time / PTBBenchmarks.BATCH_SIZE
        seq_per_sec = (
            PTBBenchmarks.BATCH_SIZE * PTBBenchmarks.ITERS / elapsed_time)
        self.logger.info(("|%s|%.4f\t|%.4f\t|%.4f") %
                         (test_name, average_time, elapsed_time, seq_per_sec))

    def _benchmark_apply(self, dev, test_name, model):
        """Only Test the forward computation.
        Args:
            dev, String: Device that on which the test is running. cpu or gpu.
            test_name, String: Name of the test.
            model, Callable: The tested model. It should be a callable object.
        """
        # TODO(Ying): use synthetic data directly generated on the device
        # instead of using real data.
        inputs = reader.train_batch(
            self.vocab,
            PTBBenchmarks.BATCH_SIZE,
            max_length=PTBBenchmarks.SEQ_LEN,
            shuffle=False,
            eager_execution=False)

        with tf.device(tu.device(dev)):
            output = model(inputs.x)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(inputs.initializer)

                for i in range(10):  #  warmup.
                    sess.run(output)

                start = time.time()
                for i in range(PTBBenchmarks.ITERS):
                    sess.run(output)

            self._report(test_name, start)

    def benchmark_whileOpLSTM_forward(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_apply(
                device,
                f"graph_whileOplstm_{device}_forward",
                m.small_model(vocab_size=len(self.vocab)))

    def benchmark_fine_grained_op_lstm_forward(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_apply(
                device, f"graph_finegrained_op_lstm_{device}_forward",
                small_model(
                    vocab_size=len(self.vocab),
                    rnn_type="fine_grained_op_lstm"))

    def benchmark_staticlstm_forward(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_apply(
                device, f"graph_staticlstm_{device}_forward",
                small_model(
                    vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_forward(self):
        self._benchmark_apply(
            "gpu", "graph_cudnnlstm_forward",
            small_model(vocab_size=len(self.vocab), rnn_type="cudnn_lstm"))

    def _benchmark_train(self, dev, test_name, model):
        """Test both forward and backward.
        Args:
            dev, String: Device that on which the test is running. cpu or gpu.
            test_name, String: Name of the test.
            model, Callable: The tested model. It should be a callable object.
        """
        # TODO(Ying): use  synthetic data directly generated on the device
        # instead of using real data.
        inputs = reader.train_batch(
            self.vocab,
            PTBBenchmarks.BATCH_SIZE,
            max_length=PTBBenchmarks.SEQ_LEN,
            shuffle=False,
            eager_execution=False)

        with tf.device(tu.device(dev)):
            loss = loss_fn(model, inputs.x, inputs.y)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            train_op = optimizer.minimize(loss)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(inputs.initializer)

                for i in range(10):  #  warmup.
                    loss_value, _ = sess.run([loss, train_op])
                    self.logger.debug(
                        "batch %d, loss_value = %.4f" % (i, loss_value))

                start = time.time()
                for i in range(PTBBenchmarks.ITERS):
                    loss_value, _ = sess.run([loss, train_op])
                    self.logger.debug(
                        "batch %d, loss_value = %.4f" % (i, loss_value))

            self._report(test_name, start)

    def benchmark_whileOpLSTM_train(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_train(
                device,
                f"graph_whileOplstm_train_{device}",
                m.small_model(vocab_size=len(self.vocab)))

    def benchmark_fine_grained_op_lstm_train(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_train(
                device, f"graph_finegrained_op_lstm_{device}_train",
                small_model(
                    vocab_size=len(self.vocab),
                    rnn_type="fine_grained_op_lstm"))

    def benchmark_staticlstm_train(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            self._benchmark_train(
                device, f"graph_staticlstm_train_{device}",
                small_model(
                    vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_train(self):
        self._benchmark_train(
            "gpu", "graph_cudnnlstm_train",
            small_model(vocab_size=len(self.vocab), rnn_type="cudnn_lstm"))


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
