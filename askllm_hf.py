from argparse import ArgumentParser
import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
from functools import partial
from tqdm import tqdm
from ask_llm.prompt import apply_prompt_template

args = ArgumentParser()
args.add_argument("--model_name", type=str, default="models/Mistral-7B-Instruct-v0.2t")
args.add_argument("--token", type=str)
args = args.parse_args()


device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    cache_dir="models/mistral-7b",
    token=args.token,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", token=args.token)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Load the dataset in streaming mode
dataset = load_from_disk("culturaxlocal")
# dataset = load_dataset("wikipedia", language="sv", date="20240520", streaming=True, cache_dir="wiki-sv",)


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


# ds = dataset["train"].take(2000)

ds = dataset
# Convert iterabledataset to dataset1
ds = Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
ds = ds.map(
    apply_prompt_template, batched=False, fn_kwargs={"tokenizer": tokenizer, "max_length": 450}
)


def word_len(example):
    example["word_count"] = len(example["text"].split(" "))
    return example


ds = ds.map(word_len)

ds = ds.select(list(range(2000)))
ds = ds.sort("word_count")

# ds = ds.remove_columns(["id", "meta"])
# Turn ds into a dataloader
dataloader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=3)


probs_450 = []
text_dicts = []

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
    ).to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1,
        do_sample=False,
        num_beams=1,
        output_logits=True,
        return_dict_in_generate=True,
    )
    for i, logits in enumerate(generated_ids["logits"][0]):
        decoded = tokenizer.decode(logits.argmax(-1))
        softmax_score = softmax(logits, dim=-1)
        probs_450.append(softmax_score[tokenizer.convert_tokens_to_ids("▁Yes")].to("cpu").item())
        prob = softmax_score[tokenizer.convert_tokens_to_ids("▁Yes")].to("cpu").item()
        text_dicts.append(
            {
                "text": " ".join(batch["text"][i].split(" ")[:450]),
                # "text_prompt": batch["text_prompt"][i],
                # "url": batch["url"][i],
                # "timestamp": batch["timestamp"][i],
                "prob": prob,
            }
        )
