from argparse import ArgumentParser
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--local_dir", type=str, default="hf_models/Mistral-7B-Instruct-v0.2")

args = parser.parse_args()

# snapshot_download(
#     args.model_name,
#     local_dir=args.local_dir,
#     max_workers=6,
# )

model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.save_pretrained(args.local_dir)
