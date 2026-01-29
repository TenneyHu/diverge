from divrag import DivRAG
from datasets import DatasetDict

def load_demo():
    prompt = "Give me one tip less than 10 words about how to improve my coding skills."
    return {"prompt": prompt}

def run_demo():
    data = load_demo()
    qid = 0
    query = data["prompt"]
    print("Prompt:", query)

    nums_answers = 10

    div = DivRAG(
        query=query,
        qid=qid,
        embed_model="text-embedding-3-small",
        llm_model="gpt-5.1",
        max_generation_num=nums_answers,
        retrieval_chunk_size=512,
        debug=True,
    )

    results = div.run()

    with open("./results/demo_output.txt", "w") as f:
        for i, res in enumerate(results):
            res = res.replace("\n", " ").replace("\t", " ").strip()
            f.write(f"{i+1}|{qid+1}:\t{res}\n")

run_demo()
