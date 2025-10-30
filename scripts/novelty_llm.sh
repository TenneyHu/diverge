#!/bin/bash

DATASET="novelty-bench"
INPUT="./data/novelty-bench.txt"
OUTPUT_DIR="./results/baselines/"
MODEL="gpt-4.1-mini"
MAX_CONCURRENCY=50

for TEMP in 0.4 0.7 1.0 1.3
do
    echo "🔥 Running with temperature = $TEMP ..."
    python ./src/baseline.py \
        --dataset $DATASET \
        --input $INPUT \
        --output_filedir $OUTPUT_DIR \
        --llm_model $MODEL \
        --max_concurrency $MAX_CONCURRENCY \
        --temperature $TEMP
done

echo "✅ All runs completed!"