#!/bin/bash

## Usage ##
# TOKEN=your_huggingface_token bash scripts/download_models.sh

# python scripts/download_model_checkpoint.py \
#     --model_name=mistralai/Mistral-7B-Instruct-v0.2 \
#     --local_dir=models/Mistral-7B-Instruct-v0.2 \
#     --token $TOKEN

# python download_model_checkpoint.py \
#     --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
#     --cache_dir=models/Meta-Llama-3-8B-Instruct \
#     --token $TOKEN

python scripts/download_model_checkpoint.py \
    --model_name=meta-llama/Meta-Llama-3-70B-Instruct \
    --local_dir=models/Meta-Llama-3-70B-Instruct \
    --token $TOKEN