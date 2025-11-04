DATASET="issue-bench"
INPUT="./data/issue_bench_search_results.json"
MODEL="gpt-5-mini"
MAX_CONCURRENCY=50


python ./src/rag.py \
    --dataset $DATASET \
    --data_path $INPUT \
    --llm_model $MODEL \
    --shuffle \
    --max_concurrency $MAX_CONCURRENCY


echo "✅ All runs completed!"
