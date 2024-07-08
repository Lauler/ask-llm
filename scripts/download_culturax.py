from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="uonlp/CulturaX")
parser.add_argument("--token", type=str, default=None)
args = parser.parse_args()

snapshot_download(
    args.dataset_name,
    repo_type="dataset",
    local_dir="/leonardo_work/EUHPC_A01_006/scandinavian-lm-temp/data/raw/culturax/sv",
    allow_patterns="sv/*",  # Wildcard pattern to only download the sv directory
    max_workers=10,
    token=args.token,
)

snapshot_download(
    args.dataset_name,
    repo_type="dataset",
    local_dir="/leonardo_work/EUHPC_A01_006/scandinavian-lm-temp/data/raw/culturax/no",
    allow_patterns="no/*",
    max_workers=10,
    token=args.token,
)

snapshot_download(
    args.dataset_name,
    repo_type="dataset",
    local_dir="/leonardo_work/EUHPC_A01_006/scandinavian-lm-temp/data/raw/culturax/da",
    allow_patterns="da/*",
    max_workers=10,
    token=args.token,
)
