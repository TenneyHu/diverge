from datasets import load_dataset

def parse_dataset(dataset: str):
    if dataset == "novelty-bench":
        ds = load_dataset("yimingzhang/novelty-bench")
        curated = ds["curated"]
        for item in curated:
            print(item['prompt'])


parse_dataset("novelty-bench")