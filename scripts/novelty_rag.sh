#!/usr/bin/env bash
set -euo pipefail

DATASET="novelty-bench"
INPUT="./data/novelty_bench_search_results.json"
MAX_CONCURRENCY=10
MODEL="gpt-5-mini"
PROVIDER="openai"

python ./src/rag.py \
    --dataset "$DATASET" \
    --data_path "$INPUT" \
    --llm_model "$MODEL" \
    --provider "$PROVIDER" \
    --max_concurrency "$MAX_CONCURRENCY" \
    --mmr 1 \


echo "✅ MMR RAG run completed!"


