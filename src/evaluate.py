
import argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def lexical_diversity(text_dict, max_n=3):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    per_input_avg_ratios = []

    for _, texts in text_dict.items():
        uniq_ratios = []

        for n in range(1, max_n + 1):
            all_ngrams = []
            for text in texts:
                tokens = text.strip().split()
                all_ngrams.extend(get_ngrams(tokens, n))
            total = len(all_ngrams)
            unique = len(set(all_ngrams))
            ratio = unique / total if total > 0 else 0
            uniq_ratios.append(ratio)

        avg_ratio = sum(uniq_ratios) / len(uniq_ratios) if uniq_ratios else 0
        per_input_avg_ratios.append(avg_ratio)
    return sum(per_input_avg_ratios) / len(per_input_avg_ratios) if per_input_avg_ratios else 0.0

def quality_score(args, queries):
    device = args.device
    model_name = args.reward_model
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    convs = []
    scores = []
    for qid, (query, answers) in queries.items():
        for answer in answers:
            conv1 = [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
            convs.append(conv1)
    for conv in convs:
        conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
        if tokenizer.bos_token is not None and conv1_formatted.startswith(tokenizer.bos_token):
            conv1_formatted = conv1_formatted[len(tokenizer.bos_token):]
        conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            score1 = rm(**conv1_tokenized).logits[0][0].item()
        scores.append(score1)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Average Quality Score: {avg_score}")
    print(f"len scores: {len(scores)}")

def semantic_diversity(model, qid_to_texts):
    diversities = []
    for texts in qid_to_texts.values():
        if len(texts) < 2:
            continue
        embeddings = model.encode(texts, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)
        upper_triangular = similarity_matrix[np.triu_indices(len(texts), k=1)]
        diversity = 1 - np.mean(upper_triangular)
        diversities.append(diversity)
    return np.mean(diversities) if diversities else 0


def main():
    parser = argparse.ArgumentParser(description="Compute Diversity")
    parser.add_argument("--query_path", type=str, default="./data/novelty-bench.txt", help="Path to query file")
    parser.add_argument("--input_path", type=str, default="./results/baselines/nolvelty-bench_gpt-5-mini.txt", help="Path to input JSON file")
    parser.add_argument("--reward_model", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B", help="Path to reward model")
    parser.add_argument("--quality", action="store_true", help="Whether to evaluate quality using reward model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for inference")
    args = parser.parse_args()
    
    

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    data = {}
    queries = {}

    with open(args.query_path, "r", encoding="utf-8") as f:
        for idx,line in enumerate(f):
            line = line.strip()
            queries[idx + 1] = (line, [])

    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            query_part, answer_part = parts
            query_id = query_part.split("|")[1].strip()
            answer = answer_part.strip()
            if query_id not in data:
                data[query_id] = []
            data[query_id].append(answer)
            query = queries.get(int(query_id), "")
            queries[int(query_id)][1].append(answer)

    if args.quality:
        quality_score(args, queries)

    diversity_score = lexical_diversity(data, max_n=3)
    print(f"Lexical Diversity Score: {diversity_score}")
    print (f"Semantic Diversity Score: {semantic_diversity(model, data)}")




if __name__ == "__main__":
    main()