import json
from pathlib import Path
from datasets import load_dataset

import tqdm

def main():
    DATASET = "samedad/mem-and-russian-jokes-dataset"
    OUTPUT = "anecd.jsonl"
    
    outp = Path(OUTPUT)
    ds = load_dataset(DATASET, split="train")
    to_delete_from = set([
        "Подписаться",
        "Анекдоты от",
        "Анекдоты и Шутки",
        "Анекдоты тут",
        "#цитата_дня",
        "@shutkaru",
        "@batin_mood",
    ])
    with outp.open("w", encoding="utf-8") as f:
        for ex in tqdm.tqdm(ds):
            prompt = ex["conversations"][0]['value'].strip()
            anecdote:str = ex["conversations"][1]['value'].strip()
            for td in to_delete_from:
                i = anecdote.find(td)
                if i >= 0:
                    anecdote = anecdote[:i].strip()
            if len(anecdote) >= 10:
                # print(anecdote)
                f.write(json.dumps({"prompt": prompt, "completion": " " + anecdote}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
