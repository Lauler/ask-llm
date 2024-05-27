import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from functools import partial
from tqdm import tqdm
from ask_llm.prompt import apply_prompt_template

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", padding_side="left"
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Load the dataset in streaming mode
dataset = load_dataset(
    "uonlp/CulturaX", "sv", streaming=True, use_auth_token=True, cache_dir="culturax"
)


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


ds = dataset["train"].take(2000)

# Convert iterabledataset to dataset
ds = Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
ds = ds.map(
    apply_prompt_template, batched=False, fn_kwargs={"tokenizer": tokenizer, "max_length": 450}
)

# Turn ds into a dataloader
dataloader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=3)

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
        add_special_tokens=False,
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
                "text": batch["text"][i],
                "url": batch["url"][i],
                "timestamp": batch["timestamp"][i],
                "prob": prob,
            }
        )

df = pd.DataFrame(text_dicts)
df.sort_values("prob", ascending=False)[1126:1127].values

# Flatten the list of tensors
probs_450 = torch.cat(probs_450).tolist()


# Find index with minimum prob
min_prob_index = probs_450.index(min(probs_450))
argmin_prob_index = torch.argmin(torch.tensor(probs_450))

df = pd.DataFrame(
    {
        "probs_450": probs_450[0:192],
    }
)

# Correlation matrix
df.corr()
