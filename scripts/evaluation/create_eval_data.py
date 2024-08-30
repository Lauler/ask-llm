import os
import json

json_files = os.listdir("eval")
# Exclude the eval_data.jsonl file
json_files = [file for file in json_files if file != "eval_data.jsonl"]

# Read jsonlines from eval
data = {}
for file in json_files:
    with open(f"eval/{file}", "r") as f:
        data[file] = f.readlines()


all_docs = []
for key in data.keys():
    docs = data[key]
    for doc in docs:
        doc = json.loads(doc)
        doc["source"] = key
        all_docs.append(doc)

bad_no = [doc for doc in all_docs if "bad_docs_no" in doc["source"]]

for doc in all_docs:
    if "good_docs" in doc["source"]:
        doc["label"] = "good"

        if "good_docs_sv" in doc["source"]:
            doc["language"] = "sv"
        else:
            doc["language"] = "no"
        doc.pop("title")

    elif "bad_docs_sv" in doc["source"]:
        doc["label"] = "bad"
        doc["language"] = "sv"
        doc["url"] = doc["metadata"]["url"]
        doc.pop("metadata")
        doc.pop("added")
    else:
        doc["label"] = "bad"
        doc["language"] = "no"
        if "paragraphs" in doc:
            doc["text"] = doc["paragraphs"][0]["text"]
            doc.pop("paragraphs")
        if "publish_date" in doc:
            doc.pop("publish_date")

        try:
            doc.pop("doc_type")
        except:
            print(doc)
            print(doc.keys())

        doc.pop("language_reported")
        doc.pop("timestamp")

    doc.pop("source")


with open("eval/eval_data.jsonl", "w", encoding="utf-8") as f:
    for doc in all_docs:
        json.dump(doc, f, ensure_ascii=False)
        f.write("\n")
