#!/bin/bash
#SBATCH --job-name=parse_json
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/parse_json-%j.out
#SBATCH --time=0-01:20:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=400GB
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=normal
#SBATCH --account=EUHPC_A02_045

echo "Present working directory is `pwd`"

PROJECT=/leonardo_work/EUHPC_A02_045/scandinavian-lm/faton/ask-llm
activate vens/ask_llm/bin/activate

python scripts/utils/parse_fineweb_json.py \
    --data_dir /leonardo_work/EUHPC_A02_045/scandinavian-lm/robin/outputs/fineweb-8b \
    --output_dir /leonardo_work/EUHPC_A02_045/scandinavian-lm/faton/outputs/fineweb-8b \
    --model_name /leonardo_work/EUHPC_A02_045/models/Meta-Llama-3.1-8B-Instruct