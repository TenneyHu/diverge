import argparse
import asyncio
import os
from pathlib import Path
from openai import AsyncOpenAI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline LLMs (Async)")
    p.add_argument("--dataset", type=str, default="novelty-bench", help="Dataset name")
    p.add_argument("--k", type=int, default=5, help="Number of trials per prompt")
    p.add_argument("--input", type=str, default="./data/novelty-bench.txt", help="Path to input file")
    p.add_argument("--output_filedir", type=str, default="./results/baselines/", help="Output directory")
    p.add_argument("--llm_model", type=str, default="gpt-5-mini", help="OpenAI model name")
    p.add_argument("--max_concurrency", type=int, default=20, help="Max concurrent requests")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM")
    return p.parse_args()


async def main_async(args):
    client = AsyncOpenAI()
    with open(args.input, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    Path(args.output_filedir).mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_filedir) / f"{args.dataset}_{args.llm_model}_{args.temperature}.txt"
    sem = asyncio.Semaphore(args.max_concurrency)

    async def one_request(idx, trial, query):
        async with sem:
            resp = await client.responses.create(
                model=args.llm_model,
                input=query,
                temperature=args.temperature
            )
            ans = resp.output_text.strip().replace("\n", " ")
            return f"{trial+1}|{idx+1}: {ans}\n"

    tasks = [
        one_request(idx, trial, query)
        for idx, query in enumerate(queries)
        for trial in range(args.k)
    ]

    results = await asyncio.gather(*tasks)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(results)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()