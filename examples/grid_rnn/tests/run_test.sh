#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

python3 test_compute_vs_data_movement.py 2>&1 | tee time.log
