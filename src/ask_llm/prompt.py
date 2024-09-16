# Create pretraining_template column in ds
def apply_prompt_template(row, prompt, tokenizer, language="Swedish", max_tokens=512):
    """
    Apply a prompt template to a row of the dataset.

    Args:
        row: Row of dataset
        tokenizer: Huggingface tokenizer
        max_length: Maximum length of source text in words.
    Returns:
        row: Row with prompt template applied in "text_prompt" column.
    """

    # Keep only first max_length tokens of text
    # text_preview = " ".join(row["text"].split()[:max_length])
    text_preview = tokenizer.decode(
        tokenizer.encode(
            row["text"],
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )
    )
    row["extract"] = text_preview
    document_context = "###\n" + text_preview + "\n###\n"

    prompt.format(language=language)

    # text = document_context + prompt
    text = prompt + document_context

    messages = [
        {
            "role": "user",
            "content": text,
        }
    ]

    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    row["text_prompt"] = text_prompt
    return row


def askllm_prompt_prefix_length(tokenizer, language="Swedish", tokenizer_type="llama"):
    prompt_first = """Does the following paragraph demarcated within ### and ###
    contain informative signal for pre-training a large-language model in {language}?
    An informative datapoint should be well-formatted, contain some
    usable knowledge of the world, and not have any autogenerated or autotranslated marketing, etc. content.

    OPTIONS:

    - yes
    - no

    """.format(
        language=language
    )

    prefix_length = tokenizer.encode(prompt_first, add_special_tokens=False)

    if tokenizer_type == "llama":
        magic_number = 47
    elif tokenizer_type == "gemma":
        magic_number = 4

    return magic_number + prefix_length


def apply_fineweb_prompt(
    example, prompt, tokenizer, max_tokens=512, language="Swedish", prompt_type="llama"
):
    # format the extract with python .format and return the prompt as dict
    extract = example["text"]
    extract_ids = tokenizer.encode(
        extract, truncation=True, max_length=max_tokens, add_special_tokens=False
    )
    extract_string = tokenizer.decode(extract_ids)
    example["extract"] = extract_string

    prompt = "\n".join([prompt[0].format(language=language), extract_string, prompt[1]])

    if prompt_type == "gemma":
        messages = [
            {"role": "user", "content": prompt},
        ]
        example["text_prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can evaluate the educational value of a web page.",
            },
            {"role": "user", "content": prompt},
        ]
        example["text_prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


def fineweb_prompt_prefix_length(prompt, tokenizer, language="Swedish", prompt_type="llama"):
    # format the extract with python .format and return the prompt as dict
    # for prefix
    # tokenize to get length
    # add number dependending on llama/gemma chat_template
    # llama 47:
    # gemma 4:
    prefix_len = len(tokenizer.encode(prompt[0].format(language), add_special_tokens=False))

    if prompt_type == "gemma":
        magic_number = 4
    else:
        magic_number = 47

    return magic_number + prefix_len
