import os
import argparse
from typing import List, Dict, Any, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from diverge_agent import PlannerAgent, SearchAgent, GenerationAgent
from tqdm import tqdm
from rag import build_or_load_index_for_query, load_index_for_query
from llama_index.core.response_synthesizers import get_response_synthesizer
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate diversified subtasks from queries")
    p.add_argument(
        "--query_path",
        type=str,
        default="./data/novelty-bench.txt",
        help="Path to txt file: one query per line",
    )
    p.add_argument(
        "--planner_output_path",
        type=str,
        default="./results/planner_subtasks.txt",
        help="Where to write line-form subtasks (qid|sid: text)",
    )
    p.add_argument(
        "--planner_template",
        type=str,
        default="./prompts/planner_template.txt",
        help="Prompt template path with ${question} and ${k}",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of diversified subtasks to request",
    )
    p.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "claude"],
        help="LLM provider",
    )
    p.add_argument(
        "--llm_model",
        type=str,
        default="gpt-5-mini",
        help="LLM model name (e.g. gpt-4o-mini, claude-3-5-haiku-20241022)",
    )

    p.add_argument(
        "--max_workers",
        type=int,
        default=20,
        help="Maximum number of concurrent planner calls",
    )

    p.add_argument(
        "--search_results_per_subtask",
        type=int,
        default=5,
        help="Number of search result pages to keep per subtask",
    )
    p.add_argument(
        "--search_min_chars",
        type=int,
        default=256,
        help="Minimum number of characters for a page to qualify",
    )
    p.add_argument(
        "--search_max_workers",
        type=int,
        default=10,
        help="Maximum concurrent search tasks",
    )
    p.add_argument(
        "--search_output_path",
        type=str,
        default="./results/planner_search.json",
        help="Where to write JSON mapping queries->subtasks->search results",
    )
    p.add_argument(
        "--reuse_planner_outputs",
        action="store_true",
        help="Whether to reuse existing planner outputs if available",
    )
    p.add_argument(
        "--reuse_retrieval_outputs",
        action="store_true",
        help="Whether to reuse existing retrieval outputs if available",
    )
    # Generation stage arguments
    p.add_argument(
        "--generation_template",
        type=str,
        default="./prompts/generation_template.txt",
        help="Template for generation with placeholders {QUESTION} {SUBTASK} {SNIPPETS}",
    )
    p.add_argument(
        "--generation_output_path",
        type=str,
        default="./results/generation.json",
        help="Where to write JSON answers per subtask",
    )
    p.add_argument(
        "--num_retrieval_snippets",
        type=int,
        default=3,
        help="Number of retrieval snippets to include in generation",
    )
    p.add_argument(
        "--snippet_char_limit",
        type=int,
        default=800,
        help="Max characters per snippet included in generation prompt",
    )
    p.add_argument(
        "--generation_max_workers",
        type=int,
        default=10,
        help="Maximum concurrent generation tasks",
    )
    p.add_argument(
        "--embed_model", 
        type=str, 
        default="text-embedding-3-large", 
        help="OpenAI embedding model")
    p.add_argument(
        "--top_k", 
        type=int, 
        default=5, 
        help="Number of top-k results to retrieve"
    )
    p.add_argument(
        "--persist_dir", 
        type=str, 
        default="../indexes", 
        help="Base dir to persist per-query indexes."
    )
    p.add_argument(
        "--build_indexes",
        action="store_true",
        help="Build or load per-query vector indexes after search phase",
    )

    return p.parse_args()


def setup_llm(provider: str, model: str):
    if provider == "openai":
        Settings.llm = OpenAI(model=model)
    elif provider == "claude":
        Settings.llm = Anthropic(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def make_llm_generate():
    def _gen(prompt: str) -> str:
        resp = Settings.llm.complete(prompt)
        return getattr(resp, "text", str(resp))

    return _gen


def load_queries(path: str) -> List[str]:
    queries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


def main():
    args = parse_args()
    
    setup_llm(args.provider, args.llm_model)
    # Set embedding model early so index construction can use it.
    Settings.embed_model = OpenAIEmbedding(model=args.embed_model)
    
    #---Planner---
    queries = load_queries(args.query_path)
    if args.reuse_planner_outputs and os.path.exists(args.planner_output_path):
        
        print(f"Reusing existing planner outputs from {args.planner_output_path}")
        structured = {}
        with open(args.planner_output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                query_part, subtask_part = parts
                qid_sid = query_part.split("|")
                if len(qid_sid) != 2:
                    continue
                qid = int(qid_sid[0].strip())
                sid = int(qid_sid[1].strip())
                subtask_text = subtask_part.strip()
                if qid not in structured:
                    structured[qid] = {"question": "", "subtasks": []}
                structured[qid]["subtasks"].append(subtask_text)
        # Convert structured to results list
        results: List[Tuple[int, str, List[str]]] = []
        for qid in sorted(structured.keys()):
            qdata = structured[qid]
            # Fill question text from queries list if available
            q_text = queries[qid - 1] if 0 <= (qid - 1) < len(queries) else ""
            results.append((qid, q_text, qdata["subtasks"]))
    else:
        planner_llm = make_llm_generate()
        planner = PlannerAgent(llm_generate=planner_llm, template_path=args.planner_template)
        if not queries:
            raise ValueError("No queries loaded from file.")

        os.makedirs(os.path.dirname(args.planner_output_path), exist_ok=True)
        structured: Dict[int, Dict[str, Any]] = {}

        def _process_one(qid: int, q: str) -> Tuple[int, str, List[str]]:
            shared_state: Dict[str, Any] = {"num_subtasks": args.k}
            subtasks = planner.step(q, shared_state)
            return qid, q, subtasks

        num_queries = len(queries)
        results: List[Tuple[int, str, List[str]] | None] = [None] * num_queries

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_qid: Dict[Any, int] = {}
            for qid, q in enumerate(queries, start=1):
                fut = executor.submit(_process_one, qid, q)
                future_to_qid[fut] = qid

            for fut in tqdm(as_completed(future_to_qid), total=num_queries, desc="Planning", unit="query"):
                qid, q, subtasks = fut.result()
                structured[qid] = {"question": q, "subtasks": subtasks}
                results[qid - 1] = (qid, q, subtasks)

        with open(args.planner_output_path, "w", encoding="utf-8") as out:
            for item in results:
                if item is None:
                    continue
                qid, q, subtasks = item
                for sid, st in enumerate(subtasks, start=1):
                    out.write(f"{qid}|{sid}: {st}\n")

    if args.reuse_retrieval_outputs and os.path.exists(args.search_output_path):
        #read existing search outputs
        print(f"Reusing existing search outputs from {args.search_output_path}")
        with open(args.search_output_path, "r", encoding="utf-8") as jsin:
            search_json = json.load(jsin)
        search_struct = search_json.get("queries", {})
    else:    
        search_agent = SearchAgent(
            num_results=args.search_results_per_subtask,
            min_chars=args.search_min_chars,
            verbose=True,
        )
        os.makedirs(os.path.dirname(args.search_output_path), exist_ok=True)

        # Structure: { query: { subtask: [ {url,text,length}, ... ] } }
        search_struct: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        def _search_one(q: str, subtask: str) -> Tuple[str, str, List[Dict[str, Any]]]:
            shared_state: Dict[str, Any] = {}
            search_agent.step(subtask, shared_state)
            res_list = []
            if "search" in shared_state and shared_state["search"]:
                item = shared_state["search"][0]
                res_list = item.get("results", [])
            return q, subtask, res_list


        search_tasks = []
        with ThreadPoolExecutor(max_workers=args.search_max_workers) as s_executor:
            for item in results:
                if item is None:
                    continue
                _qid, q, subtasks = item
                for st in subtasks:
                    fut = s_executor.submit(_search_one, q, st)
                    search_tasks.append(fut)

            for fut in tqdm(as_completed(search_tasks), total=len(search_tasks), desc="Searching", unit="subtask"):
                q, st, res_list = fut.result()
                search_struct.setdefault(q, {})[st] = res_list

    # Flat mapping for downstream consumption
    search_results: List[Dict[str, Any]] = []
    for item in results:
        if item is None:
            continue
        qid, q, subtasks = item
        for st in subtasks:
            res_list = search_struct.get("", {}).get(st, [])
            search_results.append(
                {
                    "query_id": qid,
                    "question": q,
                    "subtask": st,
                    "results": res_list,
                    "num_results": len(res_list),
                }
            )

    if args.build_indexes:
        os.makedirs(args.persist_dir, exist_ok=True)
        for q, subtask_map in search_struct.items():
            for st, pages in subtask_map.items():
                docs: List[Document] = []
                for page in pages:
                    text = page.get("text", "").strip()
                    docs.append(Document(text=text))
                build_or_load_index_for_query(query=st, docs=docs, base_persist_dir=args.persist_dir, rebuild=False)

    for qid, q, subtasks in results:
        for idx, st in enumerate(subtasks):
            index = load_index_for_query(query=st, base_persist_dir=args.persist_dir)
            if index is None:
                print(f"[RAG] No index found for subtask: {st}")
                continue

            retriever = index.as_retriever(similarity_top_k=args.top_k)
            try:
                retrieved_nodes = retriever.retrieve(st)
                print(f"[RAG] Retrieved {len(retrieved_nodes)} nodes for subtask: {st}")
                retrieved_nodes = retriever.retrieve(st)
                synthesizer = get_response_synthesizer()
                resp = synthesizer.asynthesize(query=q, nodes=retrieved_nodes)
                ans = resp.response.strip().replace("\n", " ")
                print(f"{qid}|{idx+1}: {ans}\n")

            except Exception as e:
                print(f"[RAG] Retrieval failed for subtask: {st} with error: {e}")
        break

    """
    # -------- Generation Stage --------
    os.makedirs(os.path.dirname(args.generation_output_path), exist_ok=True)
    generation_agent = GenerationAgent(
        llm_generate=gen_llm,
        template_path=args.generation_template if os.path.exists(args.generation_template) else None,
        num_snippets=args.num_retrieval_snippets,
        snippet_char_limit=args.snippet_char_limit,
    )

    # Map for quick retrieval results: search_struct[query][subtask] -> list[dict]
    generation_records: List[Dict[str, Any]] = []

    def _generate_one(q: str, subtask: str) -> Dict[str, Any]:
        retrieval_results = search_struct.get(q, {}).get(subtask, []) if isinstance(search_struct, dict) else []
        shared_state: Dict[str, Any] = {
            "question": q,
            "retrieval_results": retrieval_results,
        }
        output_list = generation_agent.step(subtask, shared_state)
        answer = output_list[0] if output_list else ""
        return {
            "question": q,
            "subtask": subtask,
            "answer": answer,
            "num_sources": len(retrieval_results),
            "source_urls": [r.get("url", "") for r in retrieval_results],
        }

    gen_tasks = []
    with ThreadPoolExecutor(max_workers=args.generation_max_workers) as g_executor:
        for item in results:
            if item is None:
                continue
            _qid, q, subtasks = item
            for st in subtasks:
                fut = g_executor.submit(_generate_one, q, st)
                gen_tasks.append(fut)
        for fut in tqdm(as_completed(gen_tasks), total=len(gen_tasks), desc="Generating", unit="subtask"):
            generation_records.append(fut.result())

    # Write generation outputs
    with open(args.generation_output_path, "w", encoding="utf-8") as gout:
        json.dump({"items": generation_records}, gout, indent=2, ensure_ascii=False)
    print(f"Generation complete. Wrote {len(generation_records)} items to {args.generation_output_path}")
    """
if __name__ == "__main__":
    main()