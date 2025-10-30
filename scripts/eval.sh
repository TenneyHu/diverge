FILELIST1=(
    "results/rag/novelty-bench_gpt-4.1-mini_0.4.txt"
    "results/rag/novelty-bench_gpt-4.1-mini_0.7.txt"
    "results/rag/novelty-bench_gpt-4.1-mini_1.0.txt"
    "results/rag/novelty-bench_gpt-4.1-mini_1.3.txt"
)

for FILE in "${FILELIST1[@]}"; do
    echo "Evaluating $FILE ..."
    python ./src/evaluate.py --input_path "./$FILE"
done