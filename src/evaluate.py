
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
    parser.add_argument("--input_path", type=str, default="./results/baselines/nolvelty-bench_gpt-5-mini.txt", help="Path to input JSON file")
    
    args = parser.parse_args()
    
    

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    data = {}
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

    diversity_score = lexical_diversity(data, max_n=3)
    print(f"Lexical Diversity Score: {diversity_score}")
    print (f"Semantic Diversity Score: {semantic_diversity(model, data)}")


if __name__ == "__main__":
    main()