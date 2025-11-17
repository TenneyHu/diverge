#!/bin/bash

DATASET="novelty-bench"
INPUT="./data/novelty-bench.txt"
OUTPUT_DIR="./results/baselines/"
MODEL="claude-haiku-4-5"
MAX_CONCURRENCY=10
PROVIDER="claude"

python ./src/baseline.py \
    --dataset $DATASET \
    --input $INPUT \
    --provider $PROVIDER \
    --output_filedir $OUTPUT_DIR \
    --llm_model $MODEL \
    --max_concurrency $MAX_CONCURRENCY 


echo "✅ All runs completed!"