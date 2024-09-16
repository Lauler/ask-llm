#!/bin/bash

#### Meta-Llama-3-70B-Instruct
DATA_DIR=$1

# MODEL_NAME="/leonardo_work/EUHPC_A02_045/models/gemma-2-27b-it"
MODEL_NAME=$2
TOKEN=$3

PROMPT="askllm_no"
# NUM_SAMPLES=12820
NUM_SAMPLES=50
PROJECT=/leonardo_work/EUHPC_A02_045/scandinavian-lm/faton/ask-llm



# Add sv-mc4-dedup-0000.jsonl to $DATA_DIR
my_file=${DATA_DIR}/eval_data.jsonl
echo "run annotation for $my_file";

#500000 / 39 =  12820 
source /leonardo_work/EUHPC_A02_045/scandinavian-lm/faton/ask-llm/venvs/ask_llm/bin/activate

# Time
NOW=$(date "+%Y.%m.%d-%H.%M.%S")
# basename of MODEL_NAME saved as MODEL
MODEL=$(basename $MODEL_NAME)
LANGUAGE="Norwegian"


srun --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=240GB \
    --gres=gpu:1 --time=0-00:30:00 --qos=normal --account=EUHPC_A02_045 \
    --output=${PROJECT}/logs/$(basename $MODEL_NAME)_${PROMPT}_${NOW}.log \
    python scripts/evaluation/askllm_hf_eval.py --model_name $MODEL_NAME --prompt $PROMPT --input_file $my_file \
    --output_dir output/eval/fineweb --language $LANGUAGE --batch_size 4 \
    --token $TOKEN &