from functools import partial
from tqdm import tqdm
from ask_llm.prompt import apply_fineweb_prompt
from ask_llm.prompt_texts import PROMPT_EN, PROMPT_SV, PROMPT_FINEWEB
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
import torch
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--model_name", type=str, default="models/gemma-2-27b-it")
args.add_argument("--prompt", type=str, default=PROMPT_SV)
args.add_argument("--data_shard", type=int, default=0)
args.add_argument("--num_samples", type=int, default=40)
args = args.parse_args()


model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    cache_dir="models/gemma-2-27b-it",
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Left pad data_shard with zeros
data_shard_pad = str(args.data_shard).zfill(5)

# dataset = load_dataset("data/culturax_sv_sharded", num_proc=12)
dataset = load_dataset(
    "parquet",
    data_files={
        "train": f"/leonardo_work/EUHPC_A01_006/scandinavian-lm-temp/data/raw/culturax/parquet/sv/sv_part_{data_shard_pad}.parquet"
    },
    split="train",
    cache_dir="/leonardo_work/EUHPC_A01_006/scandinavian-lm-temp/data/raw/culturax/arrow/sv",
)

ds = dataset.take(args.num_samples)

# Apply prompt template and save the result
ds = ds.map(
    apply_fineweb_prompt,
    batched=False,
    num_proc=8,
    fn_kwargs={
        "prompt": PROMPT_SV,
        "tokenizer": tokenizer,
        "language": "svenska",
        "max_words": 250,
        "prompt_type": "gemma",
    },
)

dataloader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=2, pin_memory=True)

generated_texts = []

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
        max_new_tokens=135,
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

ds.to_parquet(
    f"/leonardo_work/EUHPC_A01_006/scandinavian-lm-temp/data/annotations/gemma/sv-sv_prompted/{data_shard_pad}.parquet"
)
