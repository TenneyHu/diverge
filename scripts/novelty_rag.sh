DATASET="novelty-bench"
INPUT="./data/novelty_bench_search_results.json"
MAX_CONCURRENCY=10
MODEL="claude-haiku-4-5"
MAX_CONCURRENCY=10
PROVIDER="claude"

python ./src/rag.py \
    --dataset $DATASET \
    --data_path $INPUT \
    --llm_model $MODEL \
    --provider $PROVIDER \
    --search_only \
    --max_concurrency $MAX_CONCURRENCY


echo "✅ All runs completed!"


