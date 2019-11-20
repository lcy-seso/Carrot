import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only print error information.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time

import tensorflow as tf

import tf_model.data_reader as reader
import test_utils as tu
from tf_model import loss_fn
from tf_model.rnn_ptb import small_model
import tf_model.whileop_rnn as m


class PTBBenchmarks(tf.test.Benchmark):
    BATCH_SIZE = 128
    SEQ_LEN = 50
    ITERS = 50

    def __init__(self):
        self.vocab = reader.vocab()

    def _report(self, test_name, start, num_iters, dev, batch_size):
        """
        Args:
            test_name (String): Name of the test.
            start (String): Timestamp of the start time.
            num_iters (Int): Number of tested iterations.
            dev (String): Device that on which the test is running. cpu or gpu.
            batch_size (Int): Batch size.
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
                "time_elapsed": total_time
            })

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

                for _ in range(10):  #  warmup.
                    sess.run(output)

                start = time.time()
                for _ in range(PTBBenchmarks.ITERS):
                    sess.run(output)

            self._report(test_name, start, PTBBenchmarks.ITERS, tu.device(dev),
                         PTBBenchmarks.BATCH_SIZE)

    def benchmark_whileOpLSTM_cpu_forward_small(self):
        self._benchmark_apply(
            "cpu",
            "graph_whileOplstm_cpu_forward_small",
            m.small_model(vocab_size=len(self.vocab)))

    def benchmark_whileOpLSTM_gpu_forward_small(self):
        self._benchmark_apply(
            "gpu",
            "graph_whileOplstm_gpu_forward_small",
            m.small_model(vocab_size=len(self.vocab)))

    def benchmark_fine_grained_op_lstm_cpu_forward_small(self):
        self._benchmark_apply(
            "cpu", "graph_finegrained_op_lstm_cpu_forward_small",
            small_model(
                vocab_size=len(self.vocab), rnn_type="fine_grained_op_lstm"))

    def benchmark_fine_grained_op_lstm_gpu_forward_small(self):
        self._benchmark_apply(
            "gpu", "graph_finegrained_op_lstm_gpu_forward_small",
            small_model(
                vocab_size=len(self.vocab), rnn_type="fine_grained_op_lstm"))

    def benchmark_staticlstm_cpu_forward_small(self):
        self._benchmark_apply(
            "cpu", "graph_staticlstm_cpu_forward_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_staticlstm_gpu_forward_small(self):
        self._benchmark_apply(
            "gpu", "graph_staticlstm_gpu_forward_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_forward_small(self):
        self._benchmark_apply(
            "gpu", "graph_cudnnlstm_forward_small",
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
            optimizer = tf.compat.v1.train.AdamOptimizer(1.)
            train_op = optimizer.minimize(loss)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(inputs.initializer)

                for _ in range(10):  #  warmup.
                    sess.run([loss, train_op])

                start = time.time()
                for _ in range(PTBBenchmarks.ITERS):
                    sess.run([loss, train_op])

            self._report(test_name, start, PTBBenchmarks.ITERS, tu.device(dev),
                         PTBBenchmarks.BATCH_SIZE)

    def benchmark_whileOpLSTM_train_cpu_small(self):
        self._benchmark_apply(
            "cpu",
            "graph_whileOplstm_train_cpu_small",
            m.small_model(vocab_size=len(self.vocab)))

    def benchmark_whileOpLSTM_train_gpu_small(self):
        self._benchmark_apply(
            "gpu",
            "graph_whileOplstm_train_gpu_small",
            m.small_model(vocab_size=len(self.vocab)))

    def benchmark_fine_grained_op_lstm_cpu_train_small(self):
        self._benchmark_train(
            "cpu", "graph_finegrained_op_lstm_cpu_train_small",
            small_model(
                vocab_size=len(self.vocab), rnn_type="fine_grained_op_lstm"))

    def benchmark_fine_grained_op_lstm_gpu_train_small(self):
        self._benchmark_train(
            "gpu", "graph_finegrained_op_lstm_gpu_train_small",
            small_model(
                vocab_size=len(self.vocab), rnn_type="fine_grained_op_lstm"))

    def benchmark_staticlstm_train_cpu_small(self):
        self._benchmark_train(
            "cpu", "graph_staticlstm_train_cpu_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_staticlstm_train_gpu_small(self):
        self._benchmark_train(
            "gpu", "graph_staticlstm_train_gpu_small",
            small_model(vocab_size=len(self.vocab), rnn_type="static_lstm"))

    def benchmark_cudnnlstm_train_small(self):
        self._benchmark_train(
            "gpu", "graph_cudnnlstm_train_small",
            small_model(vocab_size=len(self.vocab), rnn_type="cudnn_lstm"))


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
