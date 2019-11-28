# Stacked LSTM Language Model in TensorFlow, PyTorch, and Julia

## Introduction

### Background

TensorFlow and PyTorch are two dominant deep learning toolkits. They are designed to automatically scale users' deep learning computation in the heterogeneous compute environment. For a long time, TensorFlow's declarative style is widely accepted to be more efficient than PyTorch's imperative style which also makes auto-parallelism hard to achieve optimal speed-up.

Both TensorFlow and PyTorch have released big changes recently. TensorFlow is equipped with eager execution to improve usability while PyTorch is equipped with TorchScript to bridge the gap between fast experiment and production and enable chances for *compile-time* analysis. They tend to have similar design choices that:

1.  seamlessly support imperative execution (PyTorch, TensorFlow Eager) for flexibility and declarative execution (TorchScript, TensorFlow Graph) for performance.
2.  an automatic transformation transforms users' computation between these two styles (front-end parser for TorchScript and TensorFlow's autograph)

### Challenges

Control-flow construct support is a huge struggle in mainstream deep learning frameworks.

We observed that besides the original SSA-style control-flow primitives in the back-end, TensorFlow re-design and re-implement its control-flow design by introducing a high-level control-flow construct in 2.0 to make it easy for compiler back-end like XLA to recognize control-flow then lowered to control-flow primitives.

Fine-grained operators with control-flow construct significantly improve flexibility but put burdens on efficiency. To improve performance, besides optimizing framework's execution overheads, for computation, frameworks have to apply many ad-hoc implementations in the form of monolithic operators with manually tuned kernels that group several fine-grained operators into a larger one.

Optimizations are not easy to scale in the framework. If users chose the performance, they probably would lose flexibility and vice versa.

### Why this test

RNN computations exactly fall in the challenge listed above. They are notorious for optimizations but is possible to be optimized and we know how human manually optimize them. Since deep learning computations is becoming more and more like a tensor program, when loop computation is combined with the context, many chances for optimizations are exposed.

We are going to get evidence and answers to the following questions through the test.

1.  Since PyTorch and TensorFlow both release their efforts on ***being efficient and flexible***, ***how far they have gone***?

2.  Even though TensorFlow and PyTorch tend to have the same design choices for users, ***whether there is a big performance difference between these two mainstream frameworks*** given the fact that they both have their own legacy issues.

3.  For stacked LSTM, a network that the community has much knowledge on how to optimize it, ***whether automatic optimizations implemented as optimization pass in frameworks which resembles compiler's program transformations is able to approximate the optimal vendor-specific implementation***?

    -   Cudnn LSTM\[[1,2](#Reference)\] is a highly optimized C++ implementation of stacked LSTM networks on CUDA architecture. We take it as the optimal implementation in this test.

4.  ***To inform us whether it is possible to take an existing infrastructure to do some fast experiments?***

## Test Conclusions

### 1. Performance in general

Generally, for CPU, TensorFlow graph mode is the most efficient implementation that has an obvious performance advantage over PyTorch.

For GPU, PyTorch has a not bad performance, even though it is slower than TensorFlow's graph mode.

1.  **TensorFlow Eager vs. PyTorch, two define-by-run implementations**

    PyTorch is always `1.x` faster than TensorFlow Eager, and the performance gap on GPU is larger than that on CPU.

2.  **Performance degradation from TensorFlow graph to TensorFlow Eager**

    Eager mode is `2.x` \~ `5.x` slower than graph mode. Eager execution exposes much performance burdens on GPU.

    *In the implementation, model and data are exactly the same, we just change the execution mode.*

### 2. TorchScript vs. PyTorch

-   About TorchScript

    TorchScript (short for TS) is an intermediate representation of a PyTorch model which is users' models defined by subclassinng `nn.Module`. Pure PyTorch model as a Python program could be ***incrementally*** translated into a TS program. After this, a TS program is offloaded from Python into a high-performance C++ environment, and many optimization passes (more to be rule-based) is applied to transform this TS program.

-   Notes about TorchScript

    TS is implemented as a Python decorator, but the automatic transformation from a PyTorch program to a TS program is not always smoothly. How use define its model will lead to different parsed TS programs and leads to different performance.

-   Result

    1.  The optimization passes in the back-end is rule-based, so for popular networks like LSTM, TS + PyTorch JIT accelerate the performance about `2.x` \~ `5.x` ***on GPU*** while performance on CPU ***is not obviously improved***.

        -   This result is reasonable since the optimization pass and PyTorch JIT implementation (which makes the use of CUDA's JIT APIs) mainly benefit massively paralleled accelerator GPU, not CPU.

    2.  ***Surprisingly, JITed fine-grained op LSTM has a better performance than CuDNN stacked LSTM network.***

### 2. TorchScript seems to be a more promising infrastructure

TS exempts us from implementing a parser and provides a near-source representation. This is TS's value for us, but we haven't decided whether or not to take it, or how to use it which will be determined by our goal.

## Test Environment

``` {.text}
OS: Ubuntu 16.04.2 LTS
TensorFlow version: 2.0.0-alpha0, compiled by gcc 5.0
PyTorch v1.3
CUDA Version 10.0
CUDNN Version 7.6.2
```

### CPU information

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

### GPU information

GeForce RTX 2080 Ti, Compute Capability 7.5

## Model: Stacked RNN LM on PTB dataset.

1.  `vocab_size` = 10001
2.  `embedding_dim` = 128
3.  `num_layers` = 3, 3 LSTM layers are stacked
4.  LSTM's `hidden_dim` = `output_dim` = 256
5.  pre-softmax projection's output dimension = `vocab_size` = 10001
6.  All training samples have a fixed length: `seq_len_` = 50
7.  `batch_size` = 128

## Test Results

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

    |                          | Average Time per Batch(s) | Elapsed Time(s) | Throughput<br>(sequence/s) |
    |:-------------------------|:--------------------------|:----------------|:---------------------------|
    | TensorFlow-Eager         | 0.5960                    | 29.8003         | 214.7626                   |
    | TensorFlow-Graph         | 0.2181                    | 10.9072         | 586.7664                   |
    | TensorFlow-Graph-whileop | 0.1886                    | 9.4304          | 678.6581                   |
    | PyTorch                  | 0.3483                    | 17.4164         | 367.4705                   |
    | PyTorch-JITed-LSTM       | 0.2596                    | 12.9794         | 493.0890                   |

2.  Entire Training

    |                          | Average Time per Batch(s) | Elapsed Time(s) | Throughput<br>(sequence/s) |
    |:-------------------------|:--------------------------|:----------------|:---------------------------|
    | TensorFlow-Eager         | 1.6727                    | 83.6353         | 76.5227                    |
    | TensorFlow-Graph         | 0.6942                    | 34.7123         | 184.3728                   |
    | TensorFlow-Graph-whileop | 0.1850                    | 9.2524          | 691.7107                   |
    | PyTorch                  | 1.1339                    | 56.6953         | 112.8841                   |
    | PyTorch-JITed-LSTM       | 0.4568                    | 22.8402         | 280.2074                   |

### GPU

1.  Forward Only

    |                          | Average Time per Batch(s) | Elapsed Time(s) | Throughput<br>(sequence/s) |
    |:-------------------------|:--------------------------|:----------------|:---------------------------|
    | TensorFlow-Eager         | 0.1889                    | 9.4431          | 677.7404                   |
    | TensorFlow-Graph         | 0.0515                    | 2.5746          | 2485.8505                  |
    | TensorFlow-Graph-whileop | 0.0706                    | 3.5285          | 1813.7786                  |
    | PyTorch                  | 0.0735                    | 3.6750          | 1741.5058                  |
    | PyTorch-JITed-LSTM       | 0.0206                    | 1.0318          | 6202.5344                  |

2.  Entire Training

    |                          | Average Time per Batch(s) | Elapsed Time(s) | Throughput<br>(sequence/s) |
    |:-------------------------|:--------------------------|:----------------|:---------------------------|
    | TensorFlow-Eager         | 0.4905                    | 24.5258         | 260.9500                   |
    | TensorFlow-Graph         | 0.0855                    | 4.2728          | 1497.8342                  |
    | TensorFlow-Graph-whileop | 0.0644                    | 3.2195          | 1987.8861                  |
    | PyTorch                  | 0.3897                    | 19.4860         | 328.4403                   |
    | PyTorch-JITed-LSTM       | 0.0256                    | 1.2789          | 5004.2944                  |

## Static LSTM

unroll C++ implemented LSTM Cell to the max sequence length.

``` {.python}
for depth in range(3):  # the outer loop iterates over depth
    for t in range(max_sequence_len):  # the inner loop iterates over max_sequence_length
        h, c = LSTMCell(x, (h, c))  # local scope
```

### CPU

1.  Forward Only

    |                    | Average Time per Batch(s) | Elapsed Time(s) | Throughput<br>(sequence/s) |
    |:-------------------|:--------------------------|:----------------|:---------------------------|
    | TensorFlow-Eager   | 0.4646                    | 23.2281         | 275.5281                   |
    | TensorFlow-Graph   | 0.2360                    | 11.8022         | 542.2732                   |
    | PyTorch            | 0.2406                    | 12.0316         | 531.9346                   |
    | PyTorch-JITed-LSTM | 0.2298                    | 11.4920         | 556.9071                   |

2.  Entire Training

    |                    | Average Time per Batch(s) | Elapsed Time(s) | Throughput<br>(sequence/s) |
    |:-------------------|:--------------------------|:----------------|:---------------------------|
    | TensorFlow-Eager   | 1.1522                    | 57.6109         | 111.0901                   |
    | TensorFlow-Graph   | 0.7444                    | 37.2208         | 171.9470                   |
    | PyTorch            | 1.0352                    | 51.7581         | 123.6522                   |
    | PyTorch-JITed-LSTM | 0.4726                    | 23.6304         | 270.8372                   |

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

**NOTE**

If implemented correctly, CuDNN LSTM's performance won't make much difference among frameworks. You will find in the below test TensorFlow's CuDnn LSTM model is slower than PyTorch's. The reason is these two implementations are not exactly the same.

PyTorch's current implementation invokes the CuDNN LSTM API for stacked LSTM in one kernel, but this API is buggy in TensorFlow 2.0. Currently, we do not find a way to call this optimal API, as a result, in TensorFlow implementation, LSTM is invoked for `num_layers` times, theoretically, disable some fusion and optimizations using GPU's multi-stream.

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

## Reference

1.  Appleyard, Jeremy, Tomas Kocisky, and Phil Blunsom. "[Optimizing performance of recurrent neural networks on gpus](https://arxiv.org/pdf/1604.01946.pdf)." arXiv preprint arXiv:1604.01946 (2016).
2.  [Optimizing Recurrent Neural Networks in cuDNN 5](https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/)
