import argparse
import glob
import logging
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from ask_llm.text import get_extract, parse_json_from_response

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data_dir",
    type=str,
    default="/leonardo_work/EUHPC_A02_045/scandinavian-lm/robin/outputs/fineweb-70b",
    help="Directory to recursively search which contains the model outputs (parquet)",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    default="/leonardo_work/EUHPC_A02_045/scandinavian-lm/faton/outputs/fineweb-70b",
    help="Directory to save the parsed data",
)
argparser.add_argument(
    "--model_name",
    type=str,
    default="/leonardo_work/EUHPC_A02_045/models/Meta-Llama-3.1-70B-Instruct",
    help="Model name to use for the tokenizer",
)

args = argparser.parse_args()

logging.basicConfig(
    filename="logs/parse_json_from_fineweb.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_shards = glob.glob(f"{args.data_dir}/**/*.parquet", recursive=True)


def parquet_processer(data_shard):
    """
    Process every data shard.
    Parse the JSON object from the model's response and add it to the DataFrame.
    """
    logging.info(f"Reading {data_shard}")
    df = pd.read_parquet(data_shard)
    logging.info(f"Processing {data_shard}")
    df["json"] = df.apply(parse_json_from_response, axis=1)
    df["extract"] = df.apply(get_extract, args=(tokenizer, 512), axis=1)
    df["score"] = df["json"].apply(lambda x: x["educational_score"] if x else None)
    df["score"] = pd.to_numeric(df["score"], dtype_backend="pyarrow")
    # Write to disk again
    output_path = Path(args.output_dir) / Path(data_shard).relative_to(args.data_dir)
    os.makedirs(output_path.parent, exist_ok=True)
    logging.info(f"Writing to {output_path}")
    df.to_parquet(output_path, index=False)


with mp.Pool(mp.cpu_count() - 2) as pool:
    _ = list(tqdm(pool.imap_unordered(parquet_processer, data_shards), total=len(data_shards)))
