import argparse
import torch
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
from prompt import *
import json
import asyncio
from openai import AsyncOpenAI
from logger import setup_debug_logger

logger = setup_debug_logger(
            log_dir="./logs",
            log_name="evaluate.log",
)

def semantic_diversity_openai(qid_to_texts):
    diversities = []
    client = OpenAI()
    for qid, texts in qid_to_texts.items():
        if len(texts) < 2:
            continue
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        # shape: (num_texts, dim)
        embeddings = torch.tensor(
            [d.embedding for d in response.data],
            dtype=torch.float32,
        )

        norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm, norm.t())
        idx = torch.triu_indices(len(texts), len(texts), offset=1)
        upper_triangular = similarity_matrix[idx[0], idx[1]]
        diversity = 1 - upper_triangular.mean().item()
        diversities.append(diversity)
        logger.info(f"QID: {qid}, Semantic Diversity: {diversity}")
    logger.info(f"Average Semantic Diversity: {float(np.mean(diversities)) if diversities else 0.0:.3f}")
    return float(np.mean(diversities)) if diversities else 0.0

VERDICT_TO_ID = {
    "Excellent": 5,
    "Good": 4,
    "Fair": 3,
    "Poor": 2,
    "Irrelevant": 1,
}

async def quality_score_async(args, queries, max_concurrency=20):
    """
    Async evaluation of answer quality for open-ended questions.

    Args:
        args: namespace or dict containing
              - model (model name)
              - quality_prompt (prompt template string)
        queries: dict
            { qid: (question: str, answers: List[str]) }
        max_concurrency: int, max number of concurrent API calls

    Returns:
        float: average quality score
    """
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def eval_one(qid, textid, question, answer):
        async with semaphore:
            prompt = args.quality_prompt.format(
                QUESTION=question,
                ANSWER=answer,
            )

            try:
                resp = await client.chat.completions.create(
                    model=args.quality_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                parsed = json.loads(raw)

                verdict = parsed.get("verdict", "Unsound")
                reason = parsed.get("reason", "Invalid JSON output")

            except Exception:
                verdict = "Unsound"
                reason = "Failed to parse JSON output"

            verdict_id = VERDICT_TO_ID.get(verdict, 0)
            logger.info(f"QID: {qid}, TextID: {textid}, Verdict: {verdict} ({verdict_id}), Reason: {reason}")
            return verdict_id

    tasks = []

    for qid, (question, answers) in queries.items():
        for textid, ans in enumerate(answers):
            tasks.append(
                eval_one(qid, textid, question, ans)
            )

    verdict_ids = await asyncio.gather(*tasks)

    final_score = float(np.mean(verdict_ids)) if verdict_ids else 0.0
    logger.info(f"Final Quality Score: {final_score:.3f}")
    return final_score

def parse_file(filepath, fixed_qid=None):
    qid_to_texts = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _, rest = line.split('|', 1)
                qid_part, answer = rest.split(':', 1)
                qid = int(qid_part)
                if fixed_qid is not None:
                    qid = fixed_qid
            except ValueError:
                continue
            answer = answer.strip()
            if answer == "Empty Response":
                logger.warning(f"QID: {qid} has empty response, skipping.")
                continue
            if qid not in qid_to_texts:
                qid_to_texts[qid] = []
            qid_to_texts[qid].append(answer)
    logger.info(f"Parsed {len(qid_to_texts)} QIDs from {filepath}")
    return qid_to_texts

async def extract_claims_async(
    args,
    queries,
    max_concurrency=20,
):
    """
    For each query:
      - extract claims from each answer
      - merge all claims under the same query
    """

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def extract_from_answer(qid, original_query, answer_text):
        async with semaphore:
            claims = []

            prompt = args.claim_extraction_prompt.format(
                QUESTION=original_query,
                ANSWER=answer_text,
            )

            try:
                resp = await client.chat.completions.create(
                    model=args.viewpoint_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()

                parsed = json.loads(raw)
                extracted = parsed.get("claims", [])

                if isinstance(extracted, list):
                    claims = [c.strip() for c in extracted if isinstance(c, str)]

            except Exception as e:
                # failure → empty claim list
                claims = []
                logger.error(f"QID: {qid}, Claim extraction failed: {e}")
            logger.info(f"QID: {qid}, Extracted {len(claims)} claims from answer.")
            return qid, claims

    # ---- build tasks ----
    tasks = []
    for qid, (orig_query, texts) in queries.items():
        for text in texts:
            tasks.append(
                extract_from_answer(qid, orig_query, text)
            )

    # ---- run async ----
    results = await asyncio.gather(*tasks)

    # ---- merge claims per query ----
    merged_claims = {}
    for qid, claims in results:
        if qid not in merged_claims:
            merged_claims[qid] = []
        merged_claims[qid].extend(claims)

    return merged_claims

def average_unique_claims_per_qid_embedding(
    qid_to_texts,
    threshold=0.75,
):
    client = OpenAI()
    counts = []

    for qid, texts in qid_to_texts.items():
        if not texts:
            continue

        embeddings = get_embeddings(client, texts)
        count = count_unique_claims_pairwise_with_embeddings(
            texts, embeddings, threshold
        )
        logger.info(f"QID: {qid}, Start with {len(texts)} claims. Found {count} unique claims.")
        ratio = count / len(texts)
        counts.append(ratio)
    logger.info(f"Average unique claims per QID (ratio): {sum(counts) / len(counts) if counts else 0.0:.3f}")
    return sum(counts) / len(counts) if counts else 0.0

def count_unique_claims_pairwise_with_embeddings(
    texts,
    embeddings,
    threshold=0.75,
):
    """
    texts: List[str]
    embeddings: np.ndarray, shape (n, d)
    """
    unique_indices = []

    for i in range(len(texts)):
        is_dup = False
        for j in unique_indices:
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique_indices.append(i)

    return len(unique_indices)

def cosine_sim(u, v):
    u = np.array(u)
    v = np.array(v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def get_embeddings(client, texts):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([d.embedding for d in resp.data])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quality_model", type=str, default="gpt-5", help="LLM model name")
    parser.add_argument("--viewpoint_model", type=str, default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--quality_prompt", type=str, default=quality_prompt, help="Quality evaluation prompt template")
    parser.add_argument("--claim_extraction_prompt", type=str, default=claim_extraction_prompt, help="Claim extraction prompt template")
    parser.add_argument("--query_generation_prompt", type=str, default=question_generation_prompt, help="Query generation prompt template")
    parser.add_argument("--embed_model", type=str, default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--max_qid", type=int, default=100, help="Maximum QID to process")
    parser.add_argument("--input_file", type=str, default="./results/demo_output.txt", help="Input file with generated answers")
    parser.add_argument("--output_dir", type=str, default="./results/eval/", help="Output directory for evaluation results")
    parser.add_argument("--fixed_qid", type=int, default=None, help="Force QID For debugging")
    args = parser.parse_args()

    dataset = load_from_disk("~/CLAN/DivergeRAG/data/clan_diverge_dataset")["train"]
    raw_prompts = dataset["prompt"]                    



    qid_to_texts = parse_file(args.input_file, fixed_qid=args.fixed_qid)
    
    queries = {
        qid: (raw_prompts[qid - 1], texts)
        for qid, texts in qid_to_texts.items()
        if 1 <= qid <= args.max_qid
    }
    
    qid_to_texts = {qid: texts for qid, texts in qid_to_texts.items() if 1 <= qid <= args.max_qid}

    input_name = args.input_file.split('/')[-1]

    logger.info(f"Input File: {input_name}, Prepared {len(queries)} queries for evaluation.")

    final_quality_score = asyncio.run(
        quality_score_async(args, queries, max_concurrency=30)
    )

    res = asyncio.run(
        extract_claims_async(args, queries, max_concurrency=50)
    )
    view_div = average_unique_claims_per_qid_embedding(res)
    
    semantic_div = semantic_diversity_openai(qid_to_texts)

    with open(args.output_dir + input_name + ".txt", 'w', encoding='utf-8') as f:
        f.write(f"Input File: {input_name}\t")
        f.write(f"Semantic Diversity: {semantic_div:.3f}\t")
        f.write(f"View Diversity: {view_div:.3f}\t")
        f.write(f"Quality score: {final_quality_score:.3f}\n")