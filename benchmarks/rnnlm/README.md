# Test Environment

- OS: Ubuntu 16.04.2 LTS
- TensorFlow version: 2.0.0-alpha0, compiled by gcc 5.0
- PyTorch v1.3
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

# Model: Stacked RNN LM on PTB dataset.

1. `vocab_size` = 10001
1. `embedding_dim` = 128
1. `num_layers` = 3`: 3 LSTM layers are stacked.
1. LSTM's `hidden_dim` = `output_dim` = 256
1. pre-softmax projection's output dimension = `vocab_size` = 10001
1. All training samples have a fixed length: `seq_len_` = 50
1. `batch_size` = 128

# Test Results

60 batches are run. The first 10 batches are for warmup, and the left 50 batches are timed.

Metrics:

1. elapsed time: total time of running 50 batches.
1. wall time: average time of running one batch.
1. sequence per second.

## Static LSTM

unroll C++ implemented LSTM Cell to the max sequence length.

```python
for depth in range(3):  # the outer loop iterates over depth
    for t in range(max_sequence_len):  # the inner loop iterates over max_sequence_length
        h, c = LSTMCell(x, (h, c))  # local scope
```


### CPU

||wall time (s)|elapsed time (s)|sequence per second|
|:--|:--|:--|:--|
|TF Eager (forward-only)|1.9563|97.81468|65.4298|
|TF Eager (entire-training)|5.0318|251.5906|25.43815|
|TF Graph(forward-only)|0.5834|29.1745|219.3693|
|TF Graph (entire-training)|1.6191|80.9534|79.0579|
|PyTorch (forward-only))|||
|PyTorch (entire-training)|||
|PyTroch JIT on outer scope(forward-only)|||
|PyTroch JIT on outer scope(entire-training)|||

### GPU

||wall time (s)|elapsed time (s)|sequence per second|
|:--|:--|:--|:--|
|TF Eager (forward-only)|2.2122|110.6131|57.8593|
|TF Eager (entire-training)|5.7051|285.2599|22.4357|
|TF Graph(forward-only)|0.03954|1.9772|3236.9244|
|TF Graph (entire-training)|0.05296|2.6479|2417.0494|
|PyTorch (forward-only))|||
|PyTorch (entire-training)|||
|PyTroch JIT on outer scope(forward-only)|||
|PyTroch JIT on outer scope(entire-training)|||

## CuDNN LSTM

Implement the entire LSTM network in a monolithic kernel with plenty of manual optimizations.


||wall time (s)|elapsed time (s)|sequence per second|
|:--|:--|:--|:--|
|TF-Egaer (forward-only)|0.2475|12.3766|517.1054|
|TF-Eager (entrie-training)|0.9826|49.1279|130.2721|
|TF-Graph (forward-only)|0.02954|1.4769|4333.5316|
|TF-Graph (entrie-training)|0.03214|1.6069|3982.8803|
|PyTorch (forward-only)|||
|PyTorch (entrie-training)|||
