import argparse
import asyncio
import os
from pathlib import Path
import json
import aiohttp
from openai import AsyncOpenAI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline LLMs (Async)")
    p.add_argument("--dataset", type=str, default="novelty-bench", help="Dataset name")
    p.add_argument("--k", type=int, default=5, help="Number of trials per prompt")
    p.add_argument("--input", type=str, default="./data/novelty-bench.txt", help="Path to input file")
    p.add_argument("--output_filedir", type=str, default="./results/baselines/", help="Output directory")
    p.add_argument("--llm_model", type=str, default="gpt-5-mini", help="OpenAI model name")
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "claude"], help="LLM provider to use")
    p.add_argument("--max_concurrency", type=int, default=20, help="Max concurrent requests")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature for LLM")
    return p.parse_args()


async def main_async(args):
    client = None

    if args.provider == "openai":
        client = AsyncOpenAI()
    elif args.provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required for provider=claude")
    else:
        raise RuntimeError(f"Unknown provider: {args.provider}")


    with open(args.input, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]


    Path(args.output_filedir).mkdir(parents=True, exist_ok=True)
    if args.temperature is not None:
        output_file = Path(args.output_filedir) / f"{args.dataset}_{args.llm_model}_{args.temperature}.txt"
    else:
        output_file = Path(args.output_filedir) / f"{args.dataset}_{args.llm_model}.txt"

    sem = asyncio.Semaphore(args.max_concurrency)

    async def one_request(idx, trial, query, session):
        async with sem:
            if args.provider == "openai":
                if args.temperature is not None:
                    resp = await client.responses.create(
                        model=args.llm_model,
                        input=query,
                        temperature=args.temperature,
                    )
                else:
                    resp = await client.responses.create(
                        model=args.llm_model,
                        input=query,
                    )
                ans = resp.output_text.strip().replace("\n", " ")

            else:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY"),
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                }

                payload = {
                    "model": args.llm_model,     
                    "max_tokens": 256,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": query,
                                }
                            ],
                        }
                    ],
                }

                if args.temperature is not None:
                    payload["temperature"] = args.temperature

                async with session.post(url, headers=headers, json=payload, timeout=60) as r:
                    body_text = await r.text()

                    if r.status != 200:
                        raise RuntimeError(f"Anthropic API error {r.status}: {body_text}")

                    data = json.loads(body_text)

                    ans_blocks = []
                    for block in data.get("content", []):
                        if block.get("type") == "text":
                            ans_blocks.append(block.get("text", ""))

                    ans = " ".join(ans_blocks).strip().replace("\n", " ")

        return f"{trial+1}|{idx+1}: {ans}\n"

    async with aiohttp.ClientSession() as session:
        tasks = [
            one_request(idx, trial, query, session)
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