import torch
from pydantic import BaseModel
import outlines
from tqdm import tqdm
from outlines import models, generate
from datasets import load_from_disk, Dataset
from functools import partial


"""
A recreation of the Fineweb Edu prompt, where we use an LLM (Meta-Llama-3-70B-Instruct) 
to evaluate the educational value of a web page extract.

Those annotations are then used to train a BERT model that can predict the 
educational value of a web page extract in a more compute-efficient manner.

https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
"""


@outlines.prompt
def educational_score(extract, language="Swedish"):
    """Below is an extract from a web page in {{language}}. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

    - Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
    - Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
    - Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
    - Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
    - Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

    The extract: {{extract}}

    After examining the extract:

    * First briefly justify your total score, up to 100 words.
    * Conclude with the total score of the extract.

    Answer in JSON. The JSON should be a dictionary with the keys "justification" and "total_score". The "justification" key should contain your reasoning, and "total_score" should contain the extract's score.
    """


def educational_score_hf(example, max_words=300, language="Swedish"):
    # format the extract with python .format and return the prompt as dict
    extract = example["text"]
    extract = " ".join(extract.split(" ")[0:max_words])

    prompt = """Below is an extract from a web page in {language}. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

    - Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
    - Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
    - Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
    - Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
    - Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

    The extract: {extract}

    After examining the extract:

    * First briefly justify your total score, up to 100 words.
    * Conclude with the total score of the extract.

    Answer in JSON. The JSON should be a dictionary with the keys "justification" and "total_score". The "justification" key should contain your reasoning, and "total_score" should contain the extract's score.""".format(
        extract=extract, language=language
    )

    return {
        "text_prompt": prompt,
    }


class Reasoning(BaseModel):
    justification: str
    total_score: int


model_name = "models/Meta-Llama-3-70B-Instruct"

model = models.transformers(
    model_name=model_name,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
    },
)

# Load the dataset in streaming mode
dataset = load_from_disk("data/culturaxlocal")


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


ds = dataset
# Convert iterabledataset to dataset
ds = Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
# Apply prompt template and save the result
ds = ds.map(educational_score_hf, batched=False)

dataloader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=3, pin_memory=True)

results = []
for batch in tqdm(dataloader, total=len(dataloader)):
    results.extend(generate.json(model, Reasoning)(batch["text_prompt"]))
