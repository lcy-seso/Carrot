# How to Run the Test

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
