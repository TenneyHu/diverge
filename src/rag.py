import os
import re
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any
from llama_index.core import Document, VectorStoreIndex, Settings, load_index_from_storage, StorageContext
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import argparse 
import tiktoken
import random
import asyncio
from llama_index.core.response_synthesizers import get_response_synthesizer

PRICING_INFO = {
    "gpt-5-mini": {
        "llm_prompt_per_1k": 0.000050,     
        "llm_completion_per_1k": 0.000400 
    },
    "embed": {
        "embed_per_1k": 0.00002,         
    }
}

REQUIRED_FILES = ["docstore.json"]

def slugify(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-_]+", "", text)
    return text[:max_len] if text else "q"

def get_spend_log(token_handler, model):
    """Print a summary of token usage and estimated spend."""
    prompt_tokens = getattr(token_handler, "prompt_llm_token_count", 0)
    completion_tokens = getattr(token_handler, "completion_llm_token_count", 0)
    embed_tokens = getattr(token_handler, "total_embedding_token_count", 0)
    prompt_cost_usd = (prompt_tokens / 1000) * PRICING_INFO[model]["llm_prompt_per_1k"]
    completion_cost_usd = (completion_tokens / 1000) * PRICING_INFO[model]["llm_completion_per_1k"]
    embed_cost_usd = (embed_tokens / 1000) * PRICING_INFO["embed"]["embed_per_1k"]

    cost_usd = prompt_cost_usd + completion_cost_usd + embed_cost_usd

    print("\n=== Token & Spend Summary ===")
    #prompt cost
    print(f"Prompt cost:      {prompt_cost_usd:.6f}")
    print(f"Completion cost:  {completion_cost_usd:.6f}")
    print(f"Embedding cost:   {embed_cost_usd:.6f}")
    print(f"Estimated cost:   {cost_usd:.6f}\n")



def build_or_load_index_for_query(
    query: str,
    docs: List[Document],
    base_persist_dir: str,
    rebuild: bool = False,
) -> VectorStoreIndex:
    q_dir = Path(base_persist_dir) / slugify(query)
    q_dir.mkdir(parents=True, exist_ok=True)

    index_files_exist = all((q_dir / f).exists() for f in REQUIRED_FILES)

    if index_files_exist and not rebuild:
        storage_context = StorageContext.from_defaults(persist_dir=str(q_dir))
        try:
            index = load_index_from_storage(storage_context=storage_context)
            return index
        except Exception as e:
            print(f"[Index] Failed to load existing index ({e}). Rebuilding...")
    build_ctx = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(docs, storage_context=build_ctx)
    index.storage_context.persist(persist_dir=str(q_dir))
    return index
    
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal LlamaIndex RAG from file + spend log")
    p.add_argument("--data_path", type=str, default="./data/novelty_bench_search_results.json",
                   help="Path to JSON file with {'queries': {question: [{url, text}, ...]}}")
    p.add_argument("--dataset", type=str, default="novelty-bench", help="dataset")
    p.add_argument("--k", type=int, default=5, help="Number of trials per prompt")
    p.add_argument("--top_k", type=int, default=5, help="similarity_top_k for retrieval")
    p.add_argument("--llm_model", type=str, default="gpt-5-mini", help="OpenAI Chat model")
    p.add_argument("--embed_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model")
    p.add_argument("--persist_dir", type=str, default="../indexes", help="Base dir to persist per-query indexes.")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild indexes even if persisted ones exist.")
    p.add_argument("--output_path", type=str, default="./results/rag/", help="Path to output file.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle documents before indexing.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for LLM")
    p.add_argument("--max_concurrency", type=int, default=50, help="Max in-flight LLM calls")
    return p.parse_args()


async def run_all_queries(queries, args, f):
    sem = asyncio.Semaphore(args.max_concurrency)

    async def one_trial(t, idx, query, local_nodes, synthesizer):
        async with sem: 
            resp = await synthesizer.asynthesize(query=query, nodes=local_nodes)
            ans = resp.response.strip().replace("\n", " ")
            return f"{t+1}|{idx+1}: {ans}\n"

    tasks = []
    for idx, (query, docs) in enumerate(queries.items()):
        index = build_or_load_index_for_query(
            query=query,
            docs=docs,
            base_persist_dir=args.persist_dir,
            rebuild=args.rebuild,
        )
        retriever = index.as_retriever(similarity_top_k=args.top_k)
        retrieved_nodes = retriever.retrieve(query)
        synthesizer = get_response_synthesizer()

        for t in range(args.k):
            local_nodes = list(retrieved_nodes)
            if args.shuffle:
                random.seed(t)
                random.shuffle(local_nodes)
            tasks.append(
                asyncio.create_task(
                    one_trial(t, idx, query, local_nodes, synthesizer)
                )
            )

    results = await asyncio.gather(*tasks)
    f.writelines(results)

def main():
    args = parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    Settings.llm = OpenAI(model=args.llm_model, temperature=args.temperature)
    Settings.embed_model = OpenAIEmbedding(model=args.embed_model)
    token_handler = TokenCountingHandler()
    callback_manager = CallbackManager([token_handler])

    queries: Dict[str, Any] = {}
    for query, sources in tqdm(data.get("queries", {}).items()):
        for i, s in enumerate(sources):
            text = s.get("text", "").strip()
            url = s.get("url", "").strip()
            if not text:
                continue
            
            metadata = {
                "source_url": url,
                "query_hint": query,
                "source_id": f"{query[:40]}_{i}"
            }
            queries[query] = queries.get(query, []) + [Document(text=text, metadata=metadata)]
            

    if not queries:
        raise ValueError("No documents found.")

    output_dir = args.output_path +  args.dataset + "_" + args.llm_model + "_" + str(args.temperature) + ".txt"

    if args.shuffle:
        output_dir = output_dir.replace(".txt", "_shuffled.txt")
    
    with open(output_dir, "w", encoding="utf-8") as f:
        asyncio.run(run_all_queries(queries, args, f))

    # get_spend_log(token_handler, args.llm_model)

if __name__ == "__main__":
    main()