#!/bin/bash

# Force deterministic cuBLAS
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Run your actual program, passing all arguments through
python main.py "$@"