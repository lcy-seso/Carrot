# Motivations and Goals

Given the facts that:

1.  TensorFlow and PyTorch are two of the most popular mainstream deep learning toolkits. They tend to have similar high-level design choices: seamlessly support imperative execution (via PyTorch, TensorFlow Eager mode) for flexibility and declarative execution (via PyTorch JIT, TensorFlow Graph mode) for performance.

2.  Fine-grained operators and control-flow construct significantly improve flexibility but puts many burdens on efficiency. To achieve performance, existing solutions manually group several fine-grained operators into a larger one, which means, if users chose the performance, they would lose flexibility and vice versa. The key here is how many users' DL/ML programs we can automatically optimize. Otherwise, flexibility might still be harmful to performance.

The goal of this test is twofold:

1.  illustrate the tradeoff between flexibility and performance so that we could know the space for automatic optimizations.
2.  performance differences between two of the most popular mainstream deep learning toolkit: TensorFlow and PyTorch so that to help up to make a quick decision whether it is valuable to focus on one existing infrastructure to do some experiments.

## Quick conclusions

Without digging into more implementation details, from current numbers, we temporarily conclude:

~~1. TensorFlow Eager has some strange overheads on GPU execution, which could be even slower than CPU execution. By contrast, PyTorch JIT worth considering more.~~

~~2. If considering the entire training task, TensorFlow Graph execution is the most efficient one. The performance gap between PyTorch/PyTorch and TensorFlow eager is more significant on CPU execution, while the difference becomes smaller on GPU execution.~~

~~3. Surprisingly, PyTorch/PyTorch JIT's forward computation is more efficient than TensorFlow graph mode, but when considering the whole training iteration, it becomes slower than TensorFlow graph mode. It seems that the AD implementation plus parameter updating has some more overheads.~~

~~4. Optimizing control-flow and fine-grained operators will significantly improve the performance on GPU while the performance loss is not that significant on CPU execution.~~

1.  Non-comparable performance numbers between TensorFlow and PyTorch.
2.  TensorFlow Eager is unacceptably slow in the last performance test, which is caused by build configs, but it is hard to tell which build config causes this. The build follows the official document. After using a TensorFlow 2.0 installed from pip. The number becomes normal.
3.  After simplifying PyTorch models using the most intuitive and straightforward way instead of considering code-reusing, PyTorch JIT's performance is about 5 \~ 6 times faster ***on GPU***:

Known issue:

1.  TensorFlow's CudnnLSTM implementation is not aligned with PyTorch's.
    -   PyTorch's implementation uses CudnnLSTM that implements the entire stacked LSTM network in an operator.
    -   The old version of TensorFlow has the same API, but since we haven't found a way to call this monolithic CuDNN LSTM implementation, so the current implementation has to invoke the LSTM layer `num_layer` times.

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

1.  The `TF Graph whileop-lstm` implementation implements stacked LSTM network through fine-grained operators implemented LSTM Cell and TF's symbolic [tf.while\_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop) operators.
2.  The `TF Eager`/`TF Graph` unrolls the entire stacked LSTM network and implements the unrolled network using primitive operators.

### CPU

|                                             | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:---------------------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 0.3093                     | 39.5964          | 161.6308            |
| TF Eager (entire-training)                  | 0.8080                     | 103.4290         | 61.8782             |
| TF Graph (forward-only)                     | 0.0822                     | 10.5166          | 608.5637            |
| TF Graph (entire-training)                  | 0.2668                     | 34.1561          | 187.3748            |
| TF Graph whileop-lstm (forward-only)        | 0.0728                     | 9.3235           | 686.4393            |
| TF Graph whileop-lstm (entire-training)     | 0.2559                     | 32.7506          | 195.4164            |
| PyTorch (forward-only)                      | 2.6015                     | 130.0766         | 49.2018             |
| PyTorch JIT on outer loop (forwad only)     | 2.1746                     | 108.7291         | 58.8619             |
| PyTorch (entire-training)                   | 7.4444                     | 372.2221         | 17.1940             |
| PyTorch JIT on outer loop (entire-training) | 3.9563                     | 197.8173         | 32.3531             |

### GPU

|                                             | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:---------------------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 0.1002                     | 12.8255          | 499.0055            |
| TF Eager (entire-training)                  | 0.2425                     | 31.0448          | 206.1538            |
| TF Graph (forward-only)                     | 0.0261                     | 3.3426           | 1914.6698           |
| TF Graph (entire-training)                  | 0.0370                     | 4.7342           | 1351.8558           |
| TF Graph whileop-lstm (forward-only)        | 0.0257                     | 3.2957           | 1941.9330           |
| TF Graph whileop-lstm (entire-training)     | 0.0633                     | 8.1007           | 790.0600            |
| PyTorch (forward-only)                      | 0.0837                     | 4.1857           | 1529.0294           |
| PyTorch JIT on outer loop (forwad only)     | 0.0211                     | 1.0562           | 6059.4706           |
| PyTorch (entire-training)                   | 0.4132                     | 20.6592          | 309.7896            |
| PyTorch JIT on outer loop (entire-training) | 0.0277                     | 1.3862           | 4616.9648           |

## Static LSTM

unroll C++ implemented LSTM Cell to the max sequence length.

``` {.python}
for depth in range(3):  # the outer loop iterates over depth
    for t in range(max_sequence_len):  # the inner loop iterates over max_sequence_length
        h, c = LSTMCell(x, (h, c))  # local scope
```

### CPU

|                                             | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:---------------------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 0.1863                     | 23.8429          | 268.4233            |
| TF Eager (entire-training)                  | 0.4518                     | 57.8314          | 110.6666            |
| TF Graph (forward-only)                     | 0.0906                     | 11.5916          | 552.1253            |
| TF Graph (entire-training)                  | 0.2886                     | 36.9432          | 173.2391            |
| PyTorch JIT on outer loop (forwad only)     | 2.2533                     | 112.6654         | 56.8054             |
| PyTorch (entire-training)                   | 7.6786                     | 383.9303         | 16.6697             |
| PyTorch JIT on outer loop (entire-training) | 4.5070                     | 225.3524         | 28.4000             |

### GPU

|                                             | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
|:--------------------------------------------|:---------------------------|:-----------------|:--------------------|
| TF Eager (forward-only)                     | 0.0523                     | 6.7005           | 955.1516            |
| TF Eager (entire-training)                  | 0.1069                     | 13.6853          | 467.6563            |
| TF Graph (forward-only)                     | 0.0168                     | 2.1494           | 2977.5837           |
| TF Graph (entire-training)                  | 0.0269                     | 3.4441           | 1858.2402           |
| PyTorch (forward-only)                      | 0.0211                     | 1.0529           | 6078.2238           |
| PyTorch JIT on outer loop (forwad only)     | 0.0134                     | 0.6711           | 9537.0202           |
| PyTorch (entire-training)                   | 0.0980                     | 4.9020           | 1305.5805           |
| PyTorch JIT on outer loop (entire-training) | 0.0225                     | 1.1266           | 5680.6086           |

## CuDNN LSTM

Implement the entire LSTM network in a monolithic kernel with plenty of manual optimizations.

|                            | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
|:---------------------------|:---------------------------|:-----------------|:--------------------|
| TF-Egaer (forward-only)    | 0.0076                     | 0.9667           | 6620.5735           |
| TF-Eager (entrie-training) | 0.0197                     | 2.5258           | 2533.8452           |
| TF Graph (forward-only)    | 0.0118                     | 1.5102           | 4237.8452           |
| TF Graph (entire-training) | 0.0126                     | 1.6105           | 3973.9350           |
| PyTorch (forward-only)     | 0.0088                     | 0.4380           | 14611.3166          |
| PyTorch (entire-training)  | 0.0244                     | 1.2177           | 5255.7916           |
