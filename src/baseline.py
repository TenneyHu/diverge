import argparse
from openai import OpenAI
import os

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline LLMs")
    p.add_argument("--dataset", type=str, default="nolvelty-bench", help="dataset")
    p.add_argument("--k", type=int, default=5, help="numbers of trials per prompt")
    p.add_argument("--input", type=str, default="./data/novelty-bench.txt",
                   help="Path to Input")
    p.add_argument("--output_filedir", type=str, default="./results/baselines/",
                   help="Path to Output")
    p.add_argument("--llm_model", type=str, default="gpt-5-mini", help="OpenAI Chat model")
    return p.parse_args()

def main():
    args = parse_args()
    filename = args.output_filedir + args.dataset + "_" + args.llm_model + ".txt"
    queries = []
    client = OpenAI()

    with open(args.input, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    with open(filename, 'w', encoding='utf-8') as f:
        
        for idx, query in enumerate(queries):
            for trial in range(args.k):
                response = client.responses.create(
                    model=args.llm_model,
                    input=f"{query}"
                ).output_text
                ans = response.strip().replace("\n", " ")
                f.write(f"{trial+1}|{idx+1}: {ans}\n")

if __name__ == "__main__":
    main()