# Motivations and Goals

Given the facts that:

1.  TensorFlow and PyTorch are two of the most popular mainstream deep learning toolkits. They tend to have similar high-level design choices: seamlessly support imperative execution (via PyTorch, TensorFlow Eager mode) for flexibility and declarative execution (via PyTorch JIT, TensorFlow Graph mode) for performance.

2.  Fine-grained operators and control-flow construct significantly improve flexibility but puts many burdens on efficiency. To achieve performance, existing solutions manually group several fine-grained operators into a larger one, which means, if users chose the performance, they would lose flexibility and vice versa. The key here is how many users' DL/ML programs we can automatically optimize. Otherwise, flexibility might still be harmful to performance.

The goal of this test is twofold:

1.  illustrate the tradeoff between flexibility and performance so that we could know the space for automatic optimizations.
2.  performance differences between two of the most popular mainstream deep learning toolkit: TensorFlow and PyTorch so that to help up to make a quick decision whether it is valuable to focus on one existing infrastructure to do some experiments.

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

1.  Forward Only

    |                          | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager         | 0.5960                     | 29.8003          | 214.7626            |
    | TensorFlow-Graph         | 0.2181                     | 10.9072          | 586.7664            |
    | TensorFlow-Graph-whileop | 0.1886                     | 9.4304           | 678.6581            |
    | PyTorch                  | 0.3483                     | 17.4164          | 367.4705            |
    | PyTorch-JITed-LSTM       | 0.2596                     | 12.9794          | 493.0890            |

2.  Entire Training

    |                          | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager         | 1.6727                     | 83.6353          | 76.5227             |
    | TensorFlow-Graph         | 0.6942                     | 34.7123          | 184.3728            |
    | TensorFlow-Graph-whileop | 0.1850                     | 9.2524           | 691.7107            |
    | PyTorch                  | 1.1339                     | 56.6953          | 112.8841            |
    | PyTorch-JITed-LSTM       | 0.4568                     | 22.8402          | 280.2074            |

### GPU

1.  Forward Only

    |                          | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager         | 0.1889                     | 9.4431           | 677.7404            |
    | TensorFlow-Graph         | 0.0515                     | 2.5746           | 2485.8505           |
    | TensorFlow-Graph-whileop | 0.0706                     | 3.5285           | 1813.7786           |
    | PyTorch                  | 0.0735                     | 3.6750           | 1741.5058           |
    | PyTorch-JITed-LSTM       | 0.0206                     | 1.0318           | 6202.5344           |

2.  Entire Training

    |                          | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager         | 0.4905                     | 24.5258          | 260.9500            |
    | TensorFlow-Graph         | 0.0855                     | 4.2728           | 1497.8342           |
    | TensorFlow-Graph-whileop | 0.0644                     | 3.2195           | 1987.8861           |
    | PyTorch                  | 0.3897                     | 19.4860          | 328.4403            |
    | PyTorch-JITed-LSTM       | 0.0256                     | 1.2789           | 5004.2944           |

## Static LSTM

unroll C++ implemented LSTM Cell to the max sequence length.

``` {.python}
for depth in range(3):  # the outer loop iterates over depth
    for t in range(max_sequence_len):  # the inner loop iterates over max_sequence_length
        h, c = LSTMCell(x, (h, c))  # local scope
```

### CPU

1.  Forward Only

    |                    | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager   | 0.4646                     | 23.2281          | 275.5281            |
    | TensorFlow-Graph   | 0.2360                     | 11.8022          | 542.2732            |
    | PyTorch            | 0.2406                     | 12.0316          | 531.9346            |
    | PyTorch-JITed-LSTM | 0.2298                     | 11.4920          | 556.9071            |

2.  Entire Training

    |                    | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager   | 1.1522                     | 57.6109          | 111.0901            |
    | TensorFlow-Graph   | 0.7444                     | 37.2208          | 171.9470            |
    | PyTorch            | 1.0352                     | 51.7581          | 123.6522            |
    | PyTorch-JITed-LSTM | 0.4726                     | 23.6304          | 270.8372            |

### GPU

1.  Forward Only

    |                    | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager   | 0.1324                     | 6.6212           | 966.5995            |
    | TensorFlow-Graph   | 0.0469                     | 2.3438           | 2730.6428           |
    | PyTorch            | 0.0170                     | 0.8510           | 7520.6056           |
    | PyTorch-JITed-LSTM | 0.0117                     | 0.5859           | 10922.5600          |

2.  Entire Training

    |                    | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-------------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager   | 0.2800                     | 13.9977          | 457.2177            |
    | TensorFlow-Graph   | 0.0713                     | 3.5648           | 1795.3176           |
    | PyTorch            | 0.1588                     | 7.9408           | 805.9623            |
    | PyTorch-JITed-LSTM | 0.0169                     | 0.8441           | 7582.1600           |

## CuDNN LSTM

Implement the entire LSTM network in a monolithic kernel with plenty of manual optimizations.

1.  Forward Only

    |                  | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-----------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager | 0.0203                     | 1.0155           | 6302.4602           |
    | TensorFlow-Graph | 0.0302                     | 1.5079           | 4244.3851           |
    | PyTorch          | 0.0058                     | 0.2883           | 22202.3084          |

2.  Entire Training

    |                  | Average Time per Batch (s) | Elapsed Time (s) | Sequence per Second |
    |:-----------------|:---------------------------|:-----------------|:--------------------|
    | TensorFlow-Eager | 0.0521                     | 2.6055           | 2456.3781           |
    | TensorFlow-Graph | 0.0321                     | 1.6038           | 3990.3984           |
    | PyTorch          | 0.0235                     | 1.1772           | 5436.6047           |
