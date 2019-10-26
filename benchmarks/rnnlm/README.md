# Test Environment

``` {.text}
OS: Ubuntu 16.04.2 LTS
TensorFlow version: 2.0.0-alpha0, compiled by gcc 5.0
PyTorch v1.3
CUDA Version 10.0
CUDNN Version 7.6.2
```

## CPU information

``` {.text}
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

## GPU information

GeForce RTX 2080 Ti, Compute Capability 7.5

# Model: Stacked RNN LM on PTB dataset.

1.  `vocab_size` = 10001
2.  `embedding_dim` = 128
3.  `num_layers` = 3, 3 LSTM layers are stacked
4.  LSTM's `hidden_dim` = `output_dim` = 256
5.  pre-softmax projection's output dimension = `vocab_size` = 10001
6.  All training samples have a fixed length: `seq_len_` = 50
7.  `batch_size` = 128

# Test Results

60 batches are run. The first 10 batches are for warmup, and the left 50 batches are timed.

Metrics:

1.  elapsed time: total time of running 50 batches.
2.  wall time: average time of running one batch.
3.  sequence per second.

## LSTM Network Implemented by Fine-grained Operators

In the below tests:
1. The `TF Graph whileop-lstm` implementation implements stacked LSTM network through fine-grained operators implemented LSTM Cell and TF's symbolic [tf.while\_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop) operators.
1. The `TF Eager`/`TF Graph` unrolls the entire stacked LSTM network and implements the unrolled network using primitive operators.

### CPU

|                                             | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:--------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 1.6851        | 84.2560          | 75.9590             |
| TF Eager (entire-training)                  | 5.5387        | 276.9333         | 23.1103             |
| TF Graph(forward-only)                      | 0.6452        | 32.2603          | 198.3862            |
| TF Graph(entire-training)                   | 1.3251        | 66.2531          | 96.5992             |
| TF Graph whileop-lstm(forward-only)         | 0.5243        | 26.2165          | 244.1207            |
| TF Graph whileop-lstm(entire-training)      | 0.9210        | 46.4985          | 137.6386            |
| PyTorch (forward-only))                     |               |                  |                     |
| PyTorch (entire-training)                   |               |                  |                     |
| PyTroch JIT on outer scope(forward-only)    |               |                  |                     |
| PyTroch JIT on outer scope(entire-training) |               |                  |                     |

### GPU

|                                             | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:--------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 1.7229        | 86.1453          | 74.2931             |
| TF Eager (entire-training)                  | 6.0288        | 301.4379         | 21.2316             |
| TF Graph(forward-only)                      | 0.0435        | 2.1729           | 2945.4088           |
| TF Graph (entire-training)                  | 0.0616        | 3.0802           | 2077.7791           |
| TF Graph whileop-lstm(forward-only)         | 0.0628        | 3.1379           | 2039.5771           |
| TF Graph whileop-lstm(entire-training)      | 0.1133        | 5.6687           | 1129.0042           |
| PyTorch (forward-only))                     |               |                  |                     |
| PyTorch (entire-training)                   |               |                  |                     |
| PyTroch JIT on outer scope(forward-only)    |               |                  |                     |
| PyTroch JIT on outer scope(entire-training) |               |                  |                     |

## Static LSTM

unroll C++ implemented LSTM Cell to the max sequence length.

``` {.python}
for depth in range(3):  # the outer loop iterates over depth
    for t in range(max_sequence_len):  # the inner loop iterates over max_sequence_length
        h, c = LSTMCell(x, (h, c))  # local scope
```

### CPU

|                                             | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:--------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 1.2076        | 60.3823          | 105.9912            |
| TF Eager (entire-training)                  | 3.0750        | 161.7546         | 39.5661             |
| TF Graph(forward-only)                      | 0.5834        | 29.1745          | 219.3693            |
| TF Graph (entire-training)                  | 1.6191        | 2.6479           | 2417.0494           |
| PyTorch (forward-only))                     |               |                  |                     |
| PyTorch (entire-training)                   |               |                  |                     |
| PyTroch JIT on outer scope(forward-only)    |               |                  |                     |
| PyTroch JIT on outer scope(entire-training) |               |                  |                     |

### GPU

|                                             | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:--------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 1.2839        | 64.1965          | 9.6939              |
| TF Eager (entire-training)                  | 3.2351        | 161.7546         | 39.5661             |
| TF Graph(forward-only)                      | 0.0395        | 1.9772           | 3236.9244           |
| TF Graph (entire-training)                  | 0.0530        | 2.6479           | 2417.0494           |
| PyTorch (forward-only))                     |               |                  |                     |
| PyTorch (entire-training)                   |               |                  |                     |
| PyTroch JIT on outer scope(forward-only)    |               |                  |                     |
| PyTroch JIT on outer scope(entire-training) |               |                  |                     |

## CuDNN LSTM

Implement the entire LSTM network in a monolithic kernel with plenty of manual optimizations.

|                            | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:---------------------------|:--------------|:-----------------|:--------------------|
| TF-Egaer (forward-only)    | 0.1496        | 7.4796           | 855.6626            |
| TF-Eager (entrie-training) | 0.6481        | 32.4056          | 197.4967            |
| TF-Graph (forward-only)    | 0.0295        | 1.4769           | 4333.5316           |
| TF-Graph (entrie-training) | 0.0321        | 1.6069           | 3982.8802           |
| PyTorch (forward-only)     |               |                  |                     |
| PyTorch (entrie-training)  |               |                  |                     |
