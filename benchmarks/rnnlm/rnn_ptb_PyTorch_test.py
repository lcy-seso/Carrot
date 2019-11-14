import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import logging
import unittest
from time import time

import torch

from torch.nn import LSTM
from torch.nn.functional import cross_entropy

from pt_model import data_reader as reader
import pt_model as model


class PyTorchPTBBenchmarks(unittest.TestCase):
    BATCH_SIZE = 128
    SEQ_LEN = 50
    ITERS = 50

    LOG_DEBUG = False

    def setUp(self):
        self.vocab = reader.vocab()
        self.vocab_size = len(self.vocab)
        torch.manual_seed(1234)

        self.init_logger()

    def init_logger(self):
        self.logger = logging.getLogger()

        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt=("%(levelname)s %(message)s")))
        self.logger.addHandler(handler)

        handler.setLevel(logging.DEBUG
                         if PyTorchPTBBenchmarks.LOG_DEBUG else logging.INFO)
        self.logger.setLevel(logging.DEBUG if PyTorchPTBBenchmarks.LOG_DEBUG
                             else logging.INFO)

    def _apply_forward(self, test_name, data_loader, model):
        model.train()
        for batch_idx, (x, y) in enumerate(data_loader, start=1):
            output, _ = model(x)
            if batch_idx == 10:
                break

        torch.cuda.synchronize()
        start = time()
        for batch_idx, (x, y) in enumerate(data_loader, start=1):
            output, _ = model(x)
            if batch_idx == PyTorchPTBBenchmarks.ITERS:
                elapsed_time = time() - start
                average_time = elapsed_time / PyTorchPTBBenchmarks.ITERS
                seq_per_sec = (PyTorchPTBBenchmarks.ITERS *
                               PyTorchPTBBenchmarks.BATCH_SIZE) / elapsed_time
                self.logger.info(
                    ("%s\nAverage Time per Batch\t"
                     "|Elapsed Time\t |Sequence per Second\n"
                     "%.4f\t|%.4f\t|%.4f") % (test_name, average_time,
                                              elapsed_time, seq_per_sec))
                break

    def _apply_train(self, test_name, data_loader, model, optimizer) -> float:
        def _step(batch_id, x, y):
            optimizer.zero_grad()
            output, _ = model(x)
            loss = cross_entropy(
                output.view(-1, self.vocab_size), y.reshape(-1))
            self.logger.debug(
                "batch %d, loss_value = %.4f" % (batch_id, loss.data))
            loss.backward()
            optimizer.step()

        model.train()
        for batch_idx, (x, y) in enumerate(data_loader, start=1):
            _step(batch_idx, x, y)
            if batch_idx == 10:
                break

        torch.cuda.synchronize()
        start = time()
        for batch_idx, (x, y) in enumerate(data_loader, start=1):
            _step(batch_idx, x, y)

            if batch_idx == PyTorchPTBBenchmarks.ITERS:
                elapsed_time = time() - start
                average_time = elapsed_time / PyTorchPTBBenchmarks.ITERS
                seq_per_sec = (PyTorchPTBBenchmarks.ITERS *
                               PyTorchPTBBenchmarks.BATCH_SIZE) / elapsed_time
                self.logger.info(
                    ("%s\nAverage Time per Batch\t"
                     "|Elapsed Time\t |Sequence per Second\n"
                     "%.4f\t|%.4f\t|%.4f") % (test_name, average_time,
                                              elapsed_time, seq_per_sec))
                break

    def test_fine_grained_lstm_cup_forward(self):
        device = "cpu"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="fine_grained_op_lstm",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        self._apply_forward("fine_grained_lstm_%s_forward" % (device),
                            train_loader, m)

    def test_fine_grained_lstm_gpu_forward(self):
        device = "cuda:0"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="fine_grained_op_lstm",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        self._apply_forward("fine_grained_lstm_%s_forward" % (device),
                            train_loader, m)

    def test_fine_grained_lstm_cpu_train(self):
        device = "cpu"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="fine_grained_op_lstm",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        self._apply_train("fine_grained_lstm_%s_train" % device, train_loader,
                          m, optimizer)

    def test_fine_grained_lstm_gpu_train(self):
        device = "cuda:0"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="fine_grained_op_lstm",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        self._apply_train("fine_grained_lstm_%s_train" % device, train_loader,
                          m, optimizer)

    def test_static_lstm_cpu_forward(self):
        device = "cpu"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="lstm_cell",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        self._apply_forward("static_lstm_%s_forward" % device, train_loader, m)

    def test_static_lstm_gpu_forward(self):
        device = "cuda:0"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="lstm_cell",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        self._apply_forward("static_lstm_%s_forward" % device, train_loader, m)

    def test_static_lstm_cpu_train(self):
        device = "cpu"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="lstm_cell",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        self._apply_train("static_lstm_%s_train" % device, train_loader, m,
                          optimizer)

    def test_static_lstm_gpu_train(self):
        device = "gpu"
        train_loader = reader.train_loader(
            self.vocab,
            PyTorchPTBBenchmarks.BATCH_SIZE,
            PyTorchPTBBenchmarks.SEQ_LEN,
            device=device)
        m = model.small_model(
            cell_type="lstm_cell",
            batch_size=PyTorchPTBBenchmarks.BATCH_SIZE,
            max_seq_length=PyTorchPTBBenchmarks.SEQ_LEN,
            vocab_size=self.vocab_size).to(device)
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        self._apply_train("static_lstm_%s_train" % device, train_loader, m,
                          optimizer)


if __name__ == "__main__":
    unittest.main()
