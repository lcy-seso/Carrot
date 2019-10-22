# Test Environment

- OS: Ubuntu 16.04.2 LTS
- [TensorFlow version: 2.0.0-alpha0](https://github.com/tensorflow/tensorflow/tree/r2.0), compiled by gcc 5.0
- CUDA Version 10.0
- CUDNN Version 7.6.2

- CPU information

    ```text
    Architecture:          x86_64
    CPU op-mode(s):        32-bit, 64-bit
    Byte Order:            Little Endian
    CPU(s):                12
    On-line CPU(s) list:   0-11
    Thread(s) per core:    2
    Core(s) per socket:    6
    Socket(s):             1
    NUMA node(s):          1
    Vendor ID:             GenuineIntel
    CPU family:            6
    Model:                 63
    Stepping:              2
    CPU MHz:               1578.281
    BogoMIPS:              7000.65
    Virtualization:        VT-x
    L1d cache:             32K
    L1i cache:             32K
    L2 cache:              256K
    L3 cache:              15360K
    NUMA node0 CPU(s):     0-11
    ```
- GPU information

    GeForce RTX 2080 Ti, Compute Capability 7.5

# Model information

Stacked RNN LM on PTB dataset.

1. Model topology (small model)
    1. word embedding: vocab_size 10001, embedding_dim: 128
    1. 3 stacked LSTM: hidden_dim: 256
    1. pre-softmax projection: output_dim = vocab_size = 10001
1. Every training sample in a data batch has a fixed length: 50

# Run Test

1. Run Eager mode test.

    ``` bash
    export CUDA_VISIBLE_DEVICES="0"
    python3 rnn_ptb_eager_test.py 2>&1 | tee eager_train.log
    ```

1. Run graph mode test.

    ``` bash
    export CUDA_VISIBLE_DEVICES="0"
    python3 rnn_ptb_graph_test.py 2>&1 | tee graph_train.log
    ```

# Results

Train 30 batches. The first 10 batches are for warmup.

## Static LSTM

unroll C++ implemented LSTM Cell to the max sequence length.

||Eager (s)|Graph (s)|
|:--|:--|:--|
|CPU|73.2186|45.6057|
|GPU|78.1087|12.3420|

## CuDNN LSTM

Implement the entire LSTM network in a monolithic kernel with plenty of manual optimizations.

|Eager (s)|Graph (s)|
|:--|:--|
|14.4352|2.4225|
