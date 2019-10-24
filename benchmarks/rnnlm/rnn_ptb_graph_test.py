import time
import sys

import tensorflow as tf

import tf_model.data_reader as reader
from tf_model.rnn_ptb import small_model
from tf_model import loss_fn


def train(sess, inputs, loss, train_op, n_step):
    batch_id = 1
    while True:
        try:
            loss_value, _ = sess.run([loss, train_op])
            if batch_id % 10 == 0:
                sys.stderr.write(
                    "Batch %d: Training Loss %.2f\n" % (batch_id, loss_value))

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
        return True
        import tf_model.whileop_rnn as m

        with tf.Graph().as_default():
            with tf.device("/device:CPU:0"):
                inputs = reader.train_batch(
                    self.vocab,
                    self.batch_size,
                    max_length=self.max_seq_len,
                    shuffle=False,
                    eager_execution=False)

                model = m.small_model(inputs.x, vocab_size=len(self.vocab))
                loss = loss_fn(model, inputs.x, inputs.y)
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate)
                train_op = optimizer.minimize(loss)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(inputs.initializer)

                train(sess, inputs, loss, train_op, self.WARMUP)

                self.start = time.time()
                train(sess, inputs, loss, train_op, self.TEST_BATCHES)
                print("Time elapsed: %.4f" % (time.time() - self.start))

    # def testGraphCPUStaticLSTM(self):
    #     with tf.Graph().as_default():
    #         with tf.device("/device:CPU:0"):
    #             inputs = reader.train_batch(self.vocab,
    #                                         self.batch_size,
    #                                         max_length=self.max_seq_len,
    #                                         shuffle=False,
    #                                         eager_execution=False)

    #             model = small_model(vocab_size=len(self.vocab),
    #                                 rnn_type='static_lstm')
    #             loss = loss_fn(model, inputs.x, inputs.y)
    #             optimizer = tf.compat.v1.train.AdamOptimizer(
    #                 self.learning_rate)
    #             train_op = optimizer.minimize(loss)

    #         with tf.compat.v1.Session() as sess:
    #             sess.run(tf.compat.v1.global_variables_initializer())
    #             sess.run(inputs.initializer)

    #             train(sess, inputs, loss, train_op, self.WARMUP)
    #             self.start = time.time()
    #             train(sess, inputs, loss, train_op, self.TEST_BATCHES)
    #             print("Time elapsed: %.4f" % (time.time() - self.start))

    # def testGraphGPUStaticLSTM(self):
    #     with tf.Graph().as_default():
    #         with tf.device("/device:CPU:0"):
    #             inputs = reader.train_batch(self.vocab,
    #                                         self.batch_size,
    #                                         max_length=self.max_seq_len,
    #                                         shuffle=False,
    #                                         eager_execution=False)

    #         with tf.device("/device:GPU:0"):
    #             model = small_model(vocab_size=len(self.vocab),
    #                                 rnn_type='static_lstm')
    #             loss = loss_fn(model, inputs.x, inputs.y)
    #             optimizer = tf.compat.v1.train.AdamOptimizer(
    #                 self.learning_rate)
    #             train_op = optimizer.minimize(loss)

    #         with tf.compat.v1.Session() as sess:
    #             sess.run(tf.compat.v1.global_variables_initializer())
    #             sess.run(inputs.initializer)
    #             # warm up batches.
    #             train(sess, inputs, loss, train_op, self.WARMUP)

    #             # test batches
    #             self.start = time.time()
    #             train(sess, inputs, loss, train_op, self.TEST_BATCHES)
    #             print("Time elapsed: %.4f" % (time.time() - self.start))

    def testGraphCuDNNLSTM(self):
        with tf.Graph().as_default():
            with tf.device("/device:CPU:0"):
                inputs = reader.train_batch(
                    self.vocab,
                    self.batch_size,
                    max_length=self.max_seq_len,
                    shuffle=False,
                    eager_execution=False)

            with tf.device("/device:GPU:0"):
                model = small_model(
                    vocab_size=len(self.vocab), rnn_type='cudnn_lstm')
                loss = loss_fn(model, inputs.x, inputs.y)
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate)
                train_op = optimizer.minimize(loss)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(inputs.initializer)

                train(sess, inputs, loss, train_op, self.WARMUP)
                self.start = time.time()
                train(sess, inputs, loss, train_op, self.TEST_BATCHES)
                print("Time elapsed: %.4f" % (time.time() - self.start))


class PTBBenchmarks(tf.test.Benchmark):
    BATCH_SIZE = 128
    SEQ_LEN = 50

    def _report(self, label, start, num_iters, dev, batch_size):
        pass
        wall_time = (time.time() - start) / num_iters
        dev = "cpu" if "cpu" in dev.lower() else "gpu"
        name = "%s_%s_batch_%d" % (label, dev, batch_size)
        examples_per_sec = batch_size / wall_time
        self.report_benchmark(
            iters=num_iters,
            wall_time=wall_time,
            name=name,
            extras={"examples_per_sec": examples_per_sec})

    def _benchmark_apply(self, test_name, model):
        with tf.device(device()):
            inputs = tf.ones(
                [PTBBenchmark.SEQ_LEN, PTBBenchmark.BATCH_SIZE],
                dtype=tf.int64)

            for _ in range(10):  # Warmup
                model(inputs).cpu()
            gc.collect()

            start = time.time()
            iters = 100
            for _ in range(iters):
                model(inputs).cpu()
                self._report(test_name, start, iters, device(),
                             int(sequence_batch.shape[1]))

    def benchmark_apply_small(self):
        self._benchmark_apply(
            "eager_apply_small",
            small_model(vocab_size=len(self.vocab), rnn_type='cudnn_lstm'))

    def _benchmark_train(self, label, model):
        with tf.device(device()):
            optimizer = tf.train.GradientDescentOptimizer(1.)

            def step():
                pass

            for _ in range(10):  # Warmup
                step()
            force_gpu_sync()
            gc.collect()

            start = time.time()
            iters = 100
            for _ in range(iters):
                step()
            force_gpu_sync()
            self._report()


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
