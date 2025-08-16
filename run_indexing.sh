#!/bin/bash
# File: run_indexing.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Run the indexing model: coca-clip
python -m src.indexing --indexer gpu-index-flat-l2 --extractor coca-clip --batch-size 256
echo "coca-clip completion"

# # Run the indexing model: apple-clip
python -m src.indexing --indexer gpu-index-flat-l2 --extractor apple-clip --batch-size 256
echo "apple-clip completion"

# # Run the indexing model: align
python -m src.indexing --indexer gpu-index-flat-l2 --extractor align --batch-size 256
echo "align completion"

