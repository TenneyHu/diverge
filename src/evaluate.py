
import argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from openai import OpenAI
from tqdm import tqdm


def load_deberta_tokenizer_and_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yimingzhang/deberta-v3-large-generation-similarity"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model
    
@torch.inference_mode()
async def classifier_score(prompt: str, s1: str, s2: str):
    tokenizer, model = load_deberta_tokenizer_and_model()
    input_ids = [tokenizer.cls_token_id]
    for s in [s1, s2]:
        input_ids.extend(
            tokenizer.encode(
                s,
                truncation=True,
                max_length=128,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
        prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    score = outputs["logits"].softmax(-1)[0, 1]
    return score.cpu().item()

def view_diversity(queries, k=5):
    diversities = []
    client = OpenAI()
    prompt_template = """You will be presented with {k} responses to the same prompt.
        Your task is to analyze how many distinct viewpoints or reasoning frameworks are expressed across the responses.

        Prompt:
        {input}

        Responses:
        {response_list}

        Please determine:
        1. How many *distinct viewpoints* are represented (for example, if several responses share the same reasoning or conclusion, they count as one viewpoint).
        2. Give a short explanation describing what these different viewpoints are.

        Output your response in the following format:

        Distinct Viewpoints: N
        Explanation: <one-line short summary of each viewpoint>"""

    for qid, (query, answers) in tqdm(queries.items()):
        response_list = "\n\n".join([f"Response {i+1}: {a}" for i, a in enumerate(answers)])
        prompt = prompt_template.format(k=k, input=query, response_list=response_list)

        raw_response = client.responses.create(
                model="gpt-5-mini",
                input=prompt,
            ).output_text.strip()
        try:
            response = raw_response.split("\n")[0].strip().split(":")[1].strip()
            score = int(response) 
            diversities.append(score)
        except:
            print(f"Error processing response for query {qid}: {raw_response}")

        print(raw_response)

    view_diversity_score = np.mean(diversities) if diversities else 0
    print(f"View Diversity Score: {round(view_diversity_score, 3)}")
    

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

    model_name = args.reward_model
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
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
        conv1_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token is not None and conv1_formatted.startswith(tokenizer.bos_token):
            conv1_formatted = conv1_formatted[len(tokenizer.bos_token):]
        conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt").to(args.device)
        with torch.no_grad():
            score1 = rm(**conv1_tokenized).logits[0][0].item()
        scores.append(score1)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0

    #2-digit output
    avg_score = round(avg_score, 3)
    print(f"Average Quality Score: {avg_score}")

def semantic_diversity(model, qid_to_texts):
    diversities = []
    device = next(model.parameters()).device
    
    for texts in qid_to_texts.values():
        if len(texts) < 2:
            continue
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_tensor=True)
        embeddings = embeddings.to(device)
        
        # Compute cosine similarity using PyTorch
        norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm, norm.t())
        
        # Get upper triangular part
        upper_triangular = similarity_matrix[torch.triu_indices(len(texts), len(texts), offset=1)]
        diversity = 1 - torch.mean(upper_triangular).item()
        diversities.append(diversity)
    
    return np.mean(diversities) if diversities else 0


def main():
    parser = argparse.ArgumentParser(description="Compute Diversity")
    parser.add_argument("--query_path", type=str, default="./data/issue-bench.txt", help="Path to query file")
    parser.add_argument("--input_path", type=str, default="./results/baselines/issue-bench_gpt-5-mini.txt", help="Path to input JSON file")
    parser.add_argument("--reward_model", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B", help="Path to reward model")
    parser.add_argument("--quality", action="store_true", help="Whether to evaluate quality using reward model")
    parser.add_argument("--surface_diversity", action="store_true", help="Whether to evaluate surface diversity using reward model")
    parser.add_argument("--view_diversity", action="store_true", help="Whether to evaluate view diversity using reward model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for inference")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")



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

    if args.surface_diversity:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model = model.to(device)
        diversity_score = lexical_diversity(data, max_n=3)
        diversity_score = round(diversity_score, 3)
        print(f"Lexical Diversity Score: {diversity_score}")
        print (f"Semantic Diversity Score: {round(semantic_diversity(model, data), 3)}")
    
    if args.view_diversity:
        view_diversity(queries)





if __name__ == "__main__":
    main()