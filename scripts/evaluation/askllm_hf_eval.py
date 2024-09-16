import os
from argparse import ArgumentParser
import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
from functools import partial
from tqdm import tqdm
from ask_llm.prompt import apply_prompt_template
from ask_llm.prompt_texts import (
    ASK_LLM,
    ASK_LLM2,
)
import logging
from datetime import datetime

args = ArgumentParser()
args.add_argument(
    "--model_name",
    type=str,
    default="models/Meta-Llama-3.1-8B-Instruct",
)
args.add_argument("--prompt", type=str, default="askllm_sv")
args.add_argument("--cache_dir", type=str, default="models/cache/")
args.add_argument("--output_dir", type=str, default="output/")
args.add_argument("--input_file", type=str, default="eval/eval_data.jsonl")
args.add_argument("--language", type=str, default="Swedish")
args.add_argument("--batch_size", type=int, default=8)
args.add_argument(
    "--token",
    type=str,
    default=None,
    help="HF token to use for the model/tokenizer (if private)",
)
args = args.parse_args()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    # filename=f"logs/{os.path.basename(args.model_name)}_{args.prompt}_{current_time}.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
logger.info(args)

if args.prompt == "askllm_sv" or args.prompt == "askllm_no":
    PROMPT = ASK_LLM
elif args.prompt == "askllm2_sv" or args.prompt == "askllm2_no":
    PROMPT = ASK_LLM2

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    cache_dir=args.cache_dir,
    token=args.token,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", token=args.token)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

logging.info(f"Loading dataset: {args.input_file}")
dataset = load_dataset(
    "json",
    data_files={"train": args.input_file},
    split="train",
    cache_dir="data/cache_dir",
)
ds = dataset

if args.language == "Swedish":
    ds = ds.filter(lambda x: x["language"] == "sv")
elif args.language == "Norwegian":
    ds = ds.filter(lambda x: x["language"] == "no")

logger.info(f"Applying prompts")
ds = ds.map(
    apply_prompt_template,
    batched=False,
    num_proc=4,
    fn_kwargs={
        "prompt": PROMPT,
        "tokenizer": tokenizer,
        "language": args.language,
        "max_tokens": 512,
    },
)


# ds = ds.remove_columns(["id", "meta"])
# Turn ds into a dataloader
dataloader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=3)


scores = []
# Generate with dataloader
for batch in tqdm(dataloader):
    text_prompts = batch["text_prompt"]
    model_inputs = tokenizer(
        text_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1 if "Mistral" in args.model_name else 5,  # 5 for Llama 3.1 8B
        do_sample=False,
        num_beams=1,
        output_logits=True,
        return_dict_in_generate=True,
    )
    for i, logits in enumerate(generated_ids["logits"][-1]):
        decoded = tokenizer.decode(logits.argmax(-1))
        softmax_score = softmax(logits, dim=-1)
        prob_Yes = softmax_score[tokenizer.convert_tokens_to_ids("yes")].to("cpu").item()
        prob_yes = softmax_score[tokenizer.convert_tokens_to_ids("Yes")].to("cpu").item()
        scores.append(prob_yes + prob_Yes)

# Add scores to the dataset (as str to have same type as fineweb results)
ds = ds.add_column("text_score", [str(score) for score in scores])

model_name = (
    args.model_name.split("/")[-1]
    if args.model_name[-1] != "/"
    else args.model_name.split("/")[-2]
)

ds = ds.add_column("model_name", [model_name] * len(ds))
ds = ds.add_column("prompt_type", [args.prompt] * len(ds))

file_path = os.path.join(
    args.output_dir,
    os.path.basename(args.input_file) + f".askllm.{args.language}.{model_name}.jsonl",
)

ds.to_json(
    file_path,
    force_ascii=False,
)
