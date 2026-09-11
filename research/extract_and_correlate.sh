#!/bin/bash

# A simple wrapper to build, extract and correlate

if [ -z "$EXPERIMENTS_ROOT" ]; then
    echo "Error: EXPERIMENTS_ROOT environment variable is not set."
    echo "Example: export EXPERIMENTS_ROOT=/home/ryawszn/experiments/shiro-ef"
    exit 1
fi

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: ./extract_and_correlate.sh <dataset_name>"
    echo "Example: ./extract_and_correlate.sh glove-100-angular"
    exit 1
fi

echo "=== Building extract_cv_recall ==="
mkdir -p build && cd build
cmake .. && make extract_cv_recall -j
cd ..

echo ""
echo "=== Running C++ CV Extraction ==="
./build/extract_cv_recall $DATASET
if [ $? -ne 0 ]; then
    echo "C++ extraction failed."
    exit 1
fi

echo ""
echo "=== Running Python LID Extraction and Correlation ==="
python research/extract_and_correlate.py --dataset $DATASET "${@:2}"
