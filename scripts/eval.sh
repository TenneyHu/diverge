FILELIST1=(
    "results/rag/novelty-bench_gpt-4.1-mini_0.4.txt"
    "results/rag/novelty-bench_gpt-4.1-mini_0.7.txt"
    "results/rag/novelty-bench_gpt-4.1-mini_1.0.txt"
    "results/rag/novelty-bench_gpt-4.1-mini_1.3.txt"
    "results/rag/novelty-bench_gpt-5-mini.txt"
    "results/rag/novelty-bench_gpt-5-mini_shuffled.txt"
)

FILELIST2=(
    "results/baselines/novelty-bench_gpt-4.1-mini_0.4.txt"
    "results/baselines/novelty-bench_gpt-4.1-mini_0.7.txt"
    "results/baselines/novelty-bench_gpt-4.1-mini_1.0.txt"
    "results/baselines/novelty-bench_gpt-4.1-mini_1.3.txt"
    "results/baselines/novelty-bench_gpt-5-mini.txt"
)

FILELIST11=(
    "results/rag/issue-bench_gpt-5-mini.txt"
)

FILELIST22=(
    "results/baselines/issue-bench_gpt-5-mini.txt"
)

for FILE in "${FILELIST11[@]}"; do
    echo "Evaluating $FILE ..."
    python ./src/evaluate.py --quality --input_path "./$FILE"
done