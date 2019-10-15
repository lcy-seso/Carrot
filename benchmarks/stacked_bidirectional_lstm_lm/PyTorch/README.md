# Stacked Bidirectional LSTM Language Model

This is a **PyTorch** implementation of the Stacked Bidirectional LSTM language model.

In this benchmark, we use/implement different kinds of LSTM:
- LSTM: torch.nn.LSTM
- DefaultCellLSTM: Explicit loop with torch.nn.LSTMCell
- JITDefaultCellLSTM: Explicit loop with jit(torch.nn.LSTMCell)
- FineGrainedCellLSTM: Explicit loop with FineGrainedCell
- JITFineGrainedCellLSTM: Explicit loop with jit(FineGrainedCell)

`torch.nn.LSTM` can't run with `jit` flag `True`, or it will output:
```
/pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1268: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
```

# How to use

```
usage: train.py
                [--jit]
                [--lstm {LSTM,DefaultCellLSTM,JITDefaultCellLSTM,FineGrainedCellLSTM,JITFineGrainedCellLSTM}]
                [--bidirectional]
                [--cuda]

                --num-layers NUM_LAYERS
                [--batch-size BATCH_SIZE]
                [--epoch EPOCH]
                [--lr LR]
                [--embedding-size EMBEDDING_SIZE]
                [--hidden-size HIDDEN_SIZE]

                [--seed SEED]
                [--log-interval LOG_INTERVAL]
```

For Example

```
python3 train.py --lstm=FineGrainedCellLSTM --num-layers=3 --bidirectional --cuda
```

# FineGrainedCell

$ \begin{array}{ll}
i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
c' = f * c + i * g \\
h' = o * \tanh(c') \\
\end{array} $

## Environment

- Python 3.6.5
- PyTorch 1.2.0
- CUDA 10.1

# Data
https://microsoftapc-my.sharepoint.com/:x:/g/personal/v-hocai_microsoft_com/EfF4wp5kyXxGl3fG32Wd0vwBDX1yc4KZsreujbOyl1ox8w?e=UhoRkH