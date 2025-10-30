# python ./src/evaluate.py --input_path ./results/novelty-bench_gpt-5-mini.txt
#python ./src/baseline.py --dataset issue-bench --input ./data/issue-bench.txt --output_filedir ./results/baselines/ --llm_model gpt-5-mini --max_concurrency 20

python ./src/rag.py \
    --dataset issue-bench \
    --data_path ./data/issue_bench_search_results.json \
    --llm_model gpt-5-mini 
