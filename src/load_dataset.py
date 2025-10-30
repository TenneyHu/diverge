from datasets import load_dataset
from tqdm import tqdm
import random

def parse_dataset(dataset: str):
    if dataset == "novelty-bench":
        ds = load_dataset("yimingzhang/novelty-bench")
        curated = ds["curated"]
        for item in curated:
            print(item['prompt'])
    if dataset == "issuebench":
        
        idlist = ["lmsys-963979", "wildchat-94305", "sharegpt-35467", "lmsys-132519", "lmsys-901669", 
                  "wildchat-96044", "lmsys-63964", "wildchat-357755", "lmsys-915617", "lmsys-779127", 
                  "wildchat-646443", "lmsys-455214", "sharegpt-70247", "sharegpt-31546", "lmsys-680776",
                  "wildchat-127620", "wildchat-325974", "wildchat-605049", "wildchat-44036", "sharegpt-86517",
                  "wildchat-554911",  "lmsys-808227", "lmsys-556779",  "lmsys-714922", "lmsys-392070"
                ]
        ds = load_dataset("Paul/IssueBench", "prompts", split = "prompts_full")
        topics = {}
        for item in tqdm(ds):
            prompt = item['prompt_text']
            id = item['template_id']
            topic = item['topic_id']
            polarity = item ['topic_polarity']
            if id in idlist and polarity == "neutral":
                topics[topic] = topics.get(topic, []) + [prompt]

        for topic in topics:
            prompt = random.choice(topics[topic])
            print(f"{prompt}")

parse_dataset("issuebench")