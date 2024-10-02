import json
import re

import pandas as pd


def parse_json_from_response(row):
    """
    Parse the JSON object from the model's response.
    Use this function with a pandas DataFrame and the apply method.
    """
    text = row["promptllm_answer"]
    # Between <|start_header_id|>assistant<|end_header_id|> and <|eot_id|>
    llm_response = re.search(
        r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", text, re.DOTALL
    )

    if llm_response:
        llm_response = llm_response.group(1)
        text_json = re.search(r"(\{.*?\})", llm_response, re.DOTALL)
    else:
        text_json = None

    if not text_json:
        return {"reason": None, "educational_score": None}
    else:
        text_json = text_json.group(1)

    try:
        json_text = json.loads(text_json)
        if "educational score" in json_text:
            # Standardize the key to "educational_score"
            json_text["educational_score"] = json_text.pop("educational score")
    except json.JSONDecodeError as e:
        # Decoding mostly fails when LLM uses double quotes within double quotes in the JSON.
        # In those cases, we try to extract the educational score using regex
        score = re.search(r"[\"\']educational[_ ]score[\"\']: (\d+)", text_json, re.IGNORECASE)

        if score:
            json_text = {"reason": None, "educational_score": int(score.group(1))}
        else:
            json_text = {"reason": None, "educational_score": None}
        return json_text

    if "educational_score" not in json_text:
        json_text["educational_score"] = None
    else:
        try:
            json_text["educational_score"] = int(json_text["educational_score"])
        except ValueError:
            json_text["educational_score"] = None

    return json_text


def get_extract(row, tokenizer, max_length):
    """
    Get the document extract used in the fineweb prompt.
    """
    extract = tokenizer.decode(
        tokenizer.encode(
            row["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
    ).replace("###", "")

    return extract
