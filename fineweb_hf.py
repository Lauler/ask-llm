from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
from ask_llm.prompt import apply_fineweb_prompt
from ask_llm.prompt_texts import PROMPT_EN, PROMPT_SV, PROMPT_FINEWEB
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
import torch

args = ArgumentParser()
args.add_argument("--model_name", type=str, default="models/Meta-Llama-3-70B-Instruct")
args.add_argument("--prompt", type=str, default="swedish")
args.add_argument("--data_shard", type=int, default=0)
args.add_argument("--num_samples", type=int, default=40)
args.add_argument("--batch_size", type=int, default=8)
args.add_argument("--max_new_tokens", type=int, default=150)
args.add_argument("--cache_dir", type=str, default="models/cache/")
args.add_argument("--output_dir", type=str, default="output/")

args = args.parse_args()

if args.prompt == "fineweb":
    PROMPT = PROMPT_FINEWEB
elif args.prompt == "swedish":
    PROMPT = PROMPT_SV
elif args.prompt == "english":
    PROMPT = PROMPT_EN

prompt_type = None
attn_implementation = "eager"
if "llama" in args.model_name.lower():
    prompt_type = "llama"
elif "gemma" in args.model_name.lower():
    prompt_type = "gemma"

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=args.attn_implementation,
    cache_dir=args.cache_dir,
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

# should shuffle before take
dataset = dataset.shuffle(seed=666)
ds = dataset.take(args.num_samples)

# Apply prompt template and save the result
ds = ds.map(
    apply_fineweb_prompt,
    batched=False,
    num_proc=8,
    fn_kwargs={
        "prompt": PROMPT,
        "tokenizer": tokenizer,
        "language": "svenska",
        "max_words": 250,
        "prompt_type": prompt_type,
    },
)

dataloader = torch.utils.data.DataLoader(
    ds, batch_size=args.batch_size, num_workers=2, pin_memory=True
)

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
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        num_beams=1,
        # output_logits=True,
        return_dict_in_generate=True,
    )
    # Decode the generated ids
    generated_text = tokenizer.batch_decode(
        generated_ids["sequences"], skip_special_tokens=True
    )
    generated_texts.extend(generated_text)


# Add to the dataset
ds = ds.add_column("text_score", generated_texts)

# ds.to_parquet(
#    f"/leonardo_work/EUHPC_A01_006/scandinavian-lm-temp/data/annotations/culturax/sv-sv_prompted/{data_shard_pad}.parquet"
# )
ds.to_parquet(f"{args.output_dir}/{data_shard_pad}.parquet")
