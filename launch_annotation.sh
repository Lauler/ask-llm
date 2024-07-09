#!/bin/bash

#### Meta-Llama-3-70B-Instruct
MODEL_NAME="models/Meta-Llama-3-70B-Instruct"
PROMPT="swedish"
NUM_SAMPLES=8000
PROJECT=/leonardo_work/EUHPC_D07_027/scandinavian-lm/faton/ask-llm

for i in {31..63}
do
    DATA_SHARD=$i
    srun --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=256GB \
        --gres=gpu:4 --time=0-03:20:00 --qos=normal --account=EUHPC_D07_027 \
        --output=${PROJECT}/logs/askllm_hf_llama_${PROMPT}_${DATA_SHARD}.log \
        python fineweb_hf_llama.py --model_name $MODEL_NAME --prompt $PROMPT --data_shard $DATA_SHARD --num_samples $NUM_SAMPLES &
done

# #### gemma-2-27b-it
# MODEL_NAME="models/gemma-2-27b-it"
# PROMPT="swedish"
# NUM_SAMPLES=8000
# PROJECT=/leonardo_work/EUHPC_D07_027/scandinavian-lm/faton/ask-llm

# for i in {0..63}
# do
#     DATA_SHARD=$i
#     srun --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=14 --mem=200GB \
#         --gres=gpu:2 --time=0-02:40:00 --qos=normal --account=EUHPC_D07_027 \
#         --output=${PROJECT}/logs/fineweb_hf_gemma_${PROMPT}_${DATA_SHARD}.log \
#         python fineweb_hf_gemma.py --model_name $MODEL_NAME --prompt $PROMPT --data_shard $DATA_SHARD --num_samples $NUM_SAMPLES &
# done