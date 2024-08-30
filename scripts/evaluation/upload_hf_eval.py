import os
import glob
import json
import re
import datasets


def parse_json_from_text(row):
    if "json" in row["prompt_type"]:
        text = row["text_score"]
        # Grab the final 250 words of the text
        text_snippet = text.split()[-250:]
        text_snippet = " ".join(text_snippet)
        # Grab everything between ``` and ``` (reduce false positives)
        json_text = re.findall(r"```(.*?)```", text_snippet, re.DOTALL)

        if not json_text:
            # If answer didn't use ticks, grab everything between { and } from the final 250 words
            json_text = re.findall(r"(\{.*?\})", text_snippet, re.DOTALL)
        else:
            # Grab the json object inside the ticks
            json_text = re.findall(r"(\{.*?\})", json_text[0], re.DOTALL)

        if not json_text:
            json_text = {"reason": None, "educational score": None}
            return json_text

        # Grab everything between { and } including the brackets
        json_text = re.findall(r"(\{.*?\})", json_text[0], re.DOTALL)

        if len(json_text) > 1:
            json_text = json_text[-1]
        else:
            json_text = json_text[0]

        # Parse the json
        try:
            json_text = json.loads(json_text)
        except json.JSONDecodeError as e:
            # print(e)
            json_text = {"reason": None, "educational score": None}

        return json_text

    return None


def parse_score_from_text(row):
    if "json" in row["prompt_type"]:
        # Check if the json object has an "educational score" or "educational_score" key
        if "educational score" in row["json"]:
            return row["json"]["educational score"]
        elif "educational_score" in row["json"]:
            return row["json"]["educational_score"]

    elif "no" in row["prompt_type"]:
        text = row["text_score"]
        # Grab the digit after "Pedagogisk verdi: "
        educational_score = re.findall(r"Pedagogisk verdi: (\d)", text)
        if educational_score:
            return int(educational_score[0])
    elif "sv" in row["prompt_type"]:
        text = row["text_score"]
        # Grab the digit after "Pedagogiskt värde: "
        educational_score = re.findall(r"Pedagogiskt värde: (\d)", text)
        if educational_score:
            return int(educational_score[0])

    return None


if __name__ == "__main__":
    eval_files = glob.glob("output/eval/fineweb/*")

    dataset = datasets.load_dataset(
        "json", data_files=eval_files, split="train", cache_dir="data/eval_cache"
    )

    df = dataset.to_pandas()
    # Delete "/" if at the end of the model_name
    df["model_name"] = df["model_name"].apply(lambda x: x.rstrip("/"))
    df["model_name"] = df["model_name"].apply(lambda x: x.split("/")[-1])

    df["json"] = df.apply(parse_json_from_text, axis=1)
    df["educational_score"] = df.apply(parse_score_from_text, axis=1)

    df_json = df[df["prompt_type"].str.contains("json")]
    print(
        (
            f"Total number of JSON rows: {df_json.shape[0]}",
            f"Number of rows with non-null educational_score: {df_json['educational_score'].count()}",
            f"Number of rows with null educational_score: {df_json['educational_score'].isnull().sum()}",
        )
    )

    df_non_json = df[~df["prompt_type"].str.contains("json")]
    print(
        (
            f"Total number of non-JSON rows: {df_non_json.shape[0]}",
            f"Number of rows with non-null educational_score: {df_non_json['educational_score'].count()}",
            f"Number of rows with null educational_score: {df_non_json['educational_score'].isnull().sum()}",
        )
    )

    # Groupby language, label and prompt_type, and count how many times each educational_score appears
    df_count = (
        df.groupby(["language", "label", "model_name", "educational_score"])
        .size()
        .reset_index(name="count")
    )

    # Distrubution of educational scores
    print(df_count.to_string(index=False))

    # Output df to jsonl
    df.to_json(
        "output/eval/eval_educational.jsonl", orient="records", lines=True, force_ascii=False
    )

    # dataset.to_json(
    #     "output/eval/eval_educational.jsonl", orient="records", lines=True, force_ascii=False
    # )
