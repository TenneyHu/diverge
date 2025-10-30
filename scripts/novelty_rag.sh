DATASET="novelty-bench"
INPUT="./data/novelty_bench_search_results.json"
MODEL="gpt-4.1-mini"
MAX_CONCURRENCY=50

for TEMP in 0.4 
do
    echo "🔥 Running with temperature = $TEMP ..."
    python ./src/rag.py \
    --dataset $DATASET \
    --data_path $INPUT \
    --llm_model $MODEL \
    --temperature $TEMP \
    --max_concurrency $MAX_CONCURRENCY
done

echo "✅ All runs completed!"


