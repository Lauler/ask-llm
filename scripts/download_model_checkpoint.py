from argparse import ArgumentParser
from huggingface_hub import snapshot_download

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--local_dir", type=str, default="models/Meta-Llama-3-8B-Instruct")
parser.add_argument("--token", type=str, default=None)
args = parser.parse_args()

snapshot_download(
    args.model_name,
    local_dir=args.local_dir,
    max_workers=8,
    token=args.token,
)
