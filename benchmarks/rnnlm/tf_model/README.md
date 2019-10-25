# How to Run the Test

1. Run Eager mode test.

    ``` bash
    export CUDA_VISIBLE_DEVICES="0"
    python3 rnn_ptb_eager_test.py --benchmarks=all 2>&1 | tee eager_train.log
    ```

    All the tests' names begin with `benchmark_`. You can run a single test with the below command:

    ```bash
    python3 rnn_ptb_eager_test.py --benchmarks=<test_name>
    ```

1. Run Graph mode test.

    ``` bash
    export CUDA_VISIBLE_DEVICES="0"
    python3 rnn_ptb_graph_test.py --benchmarks=all 2>&1 | tee graph_train.log
    ```

    All the tests' names begin with `benchmark_`. You can run a single test with the below command:

    ```bash
    python3 rnn_ptb_graph_test.py --benchmarks=<test_name>
    ```
