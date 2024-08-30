import glob
import datasets

eval_files = glob.glob("output/eval/fineweb/*")

dataset = datasets.load_dataset(
    "json", data_files=eval_files, split="train", cache_dir="data/eval_cache"
)

dataset.to_json(
    "output/eval/eval_educational.jsonl", orient="records", lines=True, force_ascii=False
)
