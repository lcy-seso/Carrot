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

In the below tests: 1. The `TF Graph whileop-lstm` implementation implements stacked LSTM network through fine-grained operators implemented LSTM Cell and TF's symbolic [tf.while\_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop) operators. 1. The `TF Eager`/`TF Graph` unrolls the entire stacked LSTM network and implements the unrolled network using primitive operators.

### CPU

|                                             | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:--------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 1.5050        | 85.0492          | 75.2506             |
| TF Eager (entire-training)                  | 5.2381        | 261.9030         | 24.4365             |
| TF Graph(forward-only)                      | 0.2687        | 32.2603          | 476.3282            |
| TF Graph(entire-training)                   | 0.7869        | 39.3426          | 162.6734            |
| TF Graph whileop-lstm(forward-only)         | 0.2652        | 13.2582          | 482.7189            |
| TF Graph whileop-lstm(entire-training)      | 0.8653        | 43.2652          | 147.9247            |
| PyTorch (forward-only)                      | 0.2211        | 11.0572          | 578.8104            |
| PyTorch (entire-training)                   | 1.7270        | 86.3499          | 74.1170             |
| PyTorch JIT on outer scope(forward-only)    | 0.2077        | 10.3836          | 616.3571            |
| PyTorch JIT on outer scope(entire-training) | 1.4866        | 74.3278          | 86.1050             |

### GPU

|                                             | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:--------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 1.6109        | 79.4576          | 80.5460             |
| TF Eager (entire-training)                  | 5.6337        | 281.6842         | 22.7205             |
| TF Graph(forward-only)                      | 0.0590        | 2.9487           | 2170.4379           |
| TF Graph (entire-training)                  | 0.0916        | 4.5810           | 1397.0780           |
| TF Graph whileop-lstm(forward-only)         | 0.0721        | 3.6027           | 1776.4577           |
| TF Graph whileop-lstm(entire-training)      | 0.1858        | 9.2919           | 688.7714            |
| PyTorch (forward-only)                      | 0.0285        | 1.4241           | 4493.9478           |
| PyTorch (entire-training)                   | 0.2003        | 10.0127          | 639.1888            |
| PyTorch JIT on outer scope(forward-only)    | 0.0210        | 1.0524           | 6081.4433           |
| PyTorch JIT on outer scope(entire-training) | 0.1742        | 8.7091           | 734.8663            |

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
| TF Eager (forward-only)                     | 1.2225        | 61.1254          | 104.7027            |
| TF Eager (entire-training)                  | 3.0802        | 154.0104         | 41.5556             |
| TF Graph(forward-only)                      | 0.2970        | 14.8495          | 430.9899            |
| TF Graph (entire-training)                  | 0.8478        | 42.3902          | 150.9782            |
| PyTorch (forward-only)                      | 0.2144        | 10.7207          | 596.9736            |
| PyTorch (entire-training)                   | 1.7044        | 85.2190          | 75.1006             |
| PyTorch JIT on outer scope(forward-only)    | 0.2164        | 10.8200          | 591.4970            |
| PyTorch JIT on outer scope(entire-training) | 1.3931        | 69.6566          | 91.8794             |

### GPU

|                                             | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:--------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 1.1988        | 59.9422          | 106.7695            |
| TF Eager (entire-training)                  | 3.0028        | 150.1442         | 42.6257             |
| TF Graph(forward-only)                      | 0.0437        | 2.1837           | 2930.8486           |
| TF Graph (entire-training)                  | 0.0707        | 3.5330           | 1811.4876           |
| PyTorch (forward-only)                      | 0.0284        | 1.4225           | 4499.2769           |
| PyTorch (entire-training)                   | 0.1983        | 9.9137           | 645.5730            |
| PyTorch JIT on outer scope(forward-only)    | 0.0212        | 1.0586           | 6045.9103           |
| PyTorch JIT on outer scope(entire-training) | 0.1868        | 9.3398           | 685.2425            |

## CuDNN LSTM

Implement the entire LSTM network in a monolithic kernel with plenty of manual optimizations.

|                            | Wall Time (s) | Elapsed Time (s) | Sequence per Second |
|:---------------------------|:--------------|:-----------------|:--------------------|
| TF-Egaer (forward-only)    | 0.1469        | 7.3467           | 871.1441            |
| TF-Eager (entrie-training) | 0.5271        | 26.3526          | 242.8605            |
| TF-Graph (forward-only)    | 0.0297        | 1.4842           | 4312.2252           |
| TF-Graph (entrie-training) | 0.0311        | 1.5530           | 4120.9726           |
| PyTorch (forward-only)     | 0.0085        | 0.4243           | 15082.3043          |
| PyTorch (entire-training)  | 0.0234        | 1.1675           | 5481.7606           |
