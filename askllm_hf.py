import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from functools import partial
from tqdm import tqdm
from src.ask_llm.prompt import apply_prompt_template

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", padding_side="left"
)

tokenizer.pad_token = tokenizer.eos_token

# Load the dataset in streaming mode
dataset = load_dataset(
    "uonlp/CulturaX", "sv", streaming=True, use_auth_token=True, cache_dir="culturax"
)


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


ds = dataset["train"].take(2000)

# Convert iterabledataset to dataset
ds = Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
ds = ds.map(apply_prompt_template, batched=False, fn_kwargs={"tokenizer": tokenizer})

# Turn ds into a dataloader
dataloader = torch.utils.data.DataLoader(ds, batch_size=4)

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
        output_scores=True,
        return_dict_in_generate=True,
    )
    for logits in generated_ids["scores"]:
        decoded = tokenizer.decode(logits.argmax(-1))
        softmax_score = softmax(logits, dim=-1)
        print(
            (
                f"{decoded} \n"
                f"Yes probs: {softmax_score[:, tokenizer.convert_tokens_to_ids('▁Yes')]}\n"
            )
        )
