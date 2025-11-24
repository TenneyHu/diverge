import os
import re
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any
from llama_index.core import Document, VectorStoreIndex, Settings, load_index_from_storage, StorageContext
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import argparse 
import tiktoken
import random
import asyncio
from llama_index.core.response_synthesizers import get_response_synthesizer

REQUIRED_FILES = ["docstore.json"]

def slugify(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-_]+", "", text)
    return text[:max_len] if text else "q"

def load_index_for_query(
    query: str,
    base_persist_dir: str,
):
    q_dir = Path(base_persist_dir) / slugify(query)

    if not q_dir.exists():
        return None
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(q_dir))
        index = load_index_from_storage(storage_context=storage_context)
        return index
    except Exception as e:
        print(f"[Index] Failed to load index for '{query}': {e}")
        return None

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
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(q_dir))
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
    p.add_argument("--llm_model", type=str, default="gpt-5-mini", help="LLM model name (e.g., gpt-4o-mini, claude-3-5-haiku-20241022)")
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "claude"], help="LLM provider: openai or claude (Anthropic)")
    p.add_argument("--embed_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model")
    p.add_argument("--persist_dir", type=str, default="../indexes", help="Base dir to persist per-query indexes.")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild indexes even if persisted ones exist.")
    p.add_argument("--output_path", type=str, default="./results/rag/", help="Path to output file.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle documents before indexing.")
    p.add_argument("--search_only", action="store_true", help="Only search without llms.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for LLM")
    p.add_argument("--max_concurrency", type=int, default=50, help="Max in-flight LLM calls")
    p.add_argument("--mmr", type=int, default=0, help="MMR threshold for diversity pruning")
  
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
        # Configure optional node postprocessors for retrieval
        node_postprocessors = []
        if args.mmr == 1:

            retriever = index.as_retriever(similarity_top_k=args.top_k, vector_store_query_mode="mmr")
        else:
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
    if args.search_only == False:
        if args.provider == "openai":
            if args.temperature is not None:
                Settings.llm = OpenAI(model=args.llm_model, temperature=args.temperature)
            else:
                Settings.llm = OpenAI(model=args.llm_model)
        elif args.provider == "claude":
            # Anthropic Claude via LlamaIndex
            if args.temperature is not None:
                Settings.llm = Anthropic(model=args.llm_model, temperature=args.temperature)
            else:
                Settings.llm = Anthropic(model=args.llm_model)
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")

        Settings.embed_model = OpenAIEmbedding(model=args.embed_model)
        token_handler = TokenCountingHandler()
        callback_manager = CallbackManager([token_handler])

    queries: Dict[str, Any] = {}
    for query, sources in data.get("queries", {}).items():
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
    if args.search_only:
        output_dir = args.output_path +  args.dataset + "_search_only.txt"
        with open(output_dir, "w", encoding="utf-8") as f:
            asyncio.run(run_all_queries_search_only(queries, args, f))
    else:
        if args.temperature is not None:
            output_dir = args.output_path +  args.dataset + "_" + args.llm_model + "_" + str(args.temperature) + ".txt"
        else:
            output_dir = args.output_path +  args.dataset + "_" + args.llm_model + ".txt"

        if args.shuffle:
            output_dir = output_dir.replace(".txt", "_shuffled.txt")
            
        if args.mmr > 0:
            output_dir = output_dir.replace(".txt", f"_mmr.txt")

        with open(output_dir, "w", encoding="utf-8") as f:
            asyncio.run(run_all_queries(queries, args, f))

    # get_spend_log(token_handler, args.llm_model)

if __name__ == "__main__":
    main()