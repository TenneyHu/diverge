FILELIST1=(
    "results/rag/novelty-bench_gpt-5-mini_1.0.txt"
)

QUERY="./data/novelty-bench.txt"
for FILE in "${FILELIST1[@]}"; do
    echo "Evaluating $FILE ..."
    python ./src/evaluate.py --quality \
    --surface_diversity \
    --input_path "./$FILE" \
    --query_path $QUERY
done