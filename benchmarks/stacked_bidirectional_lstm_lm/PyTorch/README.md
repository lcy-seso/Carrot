# Stacked Bidirectional LSTM Language Model

This is a **PyTorch** implementation of the Stacked Bidirectional LSTM language model.

In this benchmark, we use/implement different kinds of LSTM:
- LSTM: torch.nn.LSTM
- DefaultCellLSTM: Explicit loop with torch.nn.LSTMCell
- FineGrainedCellLSTM: Explicit loop with FineGrainedCell

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