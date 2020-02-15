# Evaluate Wavefront transformation to Grid LSTM

## Prerequisites

- Python3.6.9
- PyTorch 1.3.1
- CUDA 10.0

## Running the tests

1. Run the naive implementation.

    ```bash
    python3 grid_rnn_naive.py \
        --batch_size=2 \
        --hidden_dim=32 \
        --depth=2
    ```
1. Run the transformed codes.

    ```bash
    python3 grid_rnn_wavefront_to_inner_2_loops.py \
        --batch_size=2 \
        --hidden_dim=32 \
        --depth=2
    ```

    ```bash
    python3 grid_rnn_wavefront_to_inner_3_loops.py \
        --batch_size=2 \
        --hidden_dim=32 \
        --depth=2
    ```

    ```bash
    python3 grid_rnn_wavefront_to_all_loops.py \
        --batch_size=2 \
        --hidden_dim=32 \
        --depth=2
    ```
