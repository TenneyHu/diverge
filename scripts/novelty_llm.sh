#!/bin/bash

DATASET="novelty-bench"
INPUT="./data/novelty-bench.txt"
OUTPUT_DIR="./results/baselines/"
MODEL="gpt-4.1-mini"
MAX_CONCURRENCY=50

for MODEL in "gpt-5-nano" "gpt-5"
do
    python ./src/baseline.py \
        --dataset $DATASET \
        --input $INPUT \
        --output_filedir $OUTPUT_DIR \
        --llm_model $MODEL \
        --max_concurrency $MAX_CONCURRENCY 
done

echo "✅ All runs completed!"