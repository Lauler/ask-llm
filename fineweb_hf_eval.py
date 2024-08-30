import os
from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
from ask_llm.prompt import apply_fineweb_prompt
from ask_llm.prompt_texts import (
    PROMPT_EN,
    PROMPT_SV,
    PROMPT_FINEWEB,
    PROMPT_FINEWEB_JSON_NO,
    PROMPT_FINEWEB_JSON_SV,
    PROMPT_FINEWEB_LANG_NO,
    PROMPT_FINEWEB_LANG_SV,
    PROMPT_FINEWEB_SV,
    PROMPT_FINEWEB_NO,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
import torch
import logging

args = ArgumentParser()
args.add_argument("--model_name", type=str, default="models/Meta-Llama-3-70B-Instruct")
args.add_argument("--prompt", type=str, default="swedish")
args.add_argument("--data_shard", type=int, default=0)
args.add_argument("--num_samples", type=int, default=40)
args.add_argument("--batch_size", type=int, default=8)
args.add_argument("--max_new_tokens", type=int, default=250)
args.add_argument("--language", type=str, default="Swedish")
args.add_argument("--cache_dir", type=str, default="models/cache/")
args.add_argument("--output_dir", type=str, default="output/")
args.add_argument("--input_file", type=str, required=True)
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(args)

if args.prompt == "fineweb":
    PROMPT = PROMPT_FINEWEB
elif args.prompt == "swedish":
    PROMPT = PROMPT_SV
elif args.prompt == "english":
    PROMPT = PROMPT_EN
elif args.prompt == "fineweb_json_no":
    PROMPT = PROMPT_FINEWEB_JSON_NO
elif args.prompt == "fineweb_json_sv":
    PROMPT = PROMPT_FINEWEB_JSON_SV
elif args.prompt == "fineweb_sv":
    PROMPT = PROMPT_FINEWEB_SV
elif args.prompt == "fineweb_no":
    PROMPT = PROMPT_FINEWEB_NO
elif args.prompt == "fineweb_lang_no":
    PROMPT = PROMPT_FINEWEB_LANG_NO
elif args.prompt == "fineweb_lang_sv":
    PROMPT = PROMPT_FINEWEB_LANG_SV

model_type = None
attn_implementation = "flash_attention_2"
if "llama" in args.model_name.lower():
    if "8b" in args.model_name.lower():
        model_type = "llama8b"
    elif "70b" in args.model_name.lower():
        model_type = "llama70b"
elif "gemma" in args.model_name.lower():
    model_type = "gemma"

logging.info(f"Loading model: {args.model_name}")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=attn_implementation,
    cache_dir=args.cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


# dataset = load_dataset("data/culturax_sv_sharded", num_proc=12)
logging.info(f"Loading dataset: {args.input_file}")
dataset = load_dataset(
    "json",
    data_files={"train": args.input_file},
    split="train",
    cache_dir="data/cache_dir",
)

ds = dataset

# Filter for language in the dataset
if "sv" in args.prompt:
    ds = ds.filter(lambda x: x["language"] == "sv")
elif "no" in args.prompt:
    ds = ds.filter(lambda x: x["language"] == "no")

# Apply prompt template and save the result
logging.info(f"Applying prompts")
ds = ds.map(
    apply_fineweb_prompt,
    batched=False,
    num_proc=8,
    fn_kwargs={
        "prompt": PROMPT,
        "tokenizer": tokenizer,
        "language": args.language,
        "max_tokens": 512,
        "prompt_type": model_type,
    },
)

logging.info(f"Example: {ds[0]}")

logging.info(f"create dataloader")
dataloader = torch.utils.data.DataLoader(
    ds, batch_size=args.batch_size, num_workers=2, pin_memory=True
)

generated_texts = []

# Generate with dataloader
logging.info(f"annotate batches")
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
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        num_beams=1,
        # output_logits=True,
        return_dict_in_generate=True,
    )
    # Decode the generated ids
    generated_text = tokenizer.batch_decode(generated_ids["sequences"], skip_special_tokens=True)
    generated_texts.extend(generated_text)


# Add to the dataset
ds = ds.add_column("text_score", generated_texts)
# ds.to_parquet(os.path.join(args.output_dir, os.path.basename(args.input_file) + f"{prompt_type}.{args.prompt}.parquet"))

ds = ds.add_column("model_name", [args.model_name] * len(ds))
ds = ds.add_column("prompt_type", [args.prompt] * len(ds))

ds.to_json(
    os.path.join(
        args.output_dir, os.path.basename(args.input_file) + f"{model_type}.{args.prompt}.jsonl"
    ),
    force_ascii=False,
)
