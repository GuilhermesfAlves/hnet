#!/bin/bash

#SBATCH --job-name=hnet-tokenizers  # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 1                   # CPUs por task
#SBATCH --mem=32G              # Memória total
#SBATCH -o ./tkn_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./tkn_%j.err     # Arquivo de erro (adiciona job id %j)

srun python compare_tokenizers.py \
    --dataset-name uonlp/CulturaX \
    --config-name pt \
    --split "train" \
    --json-out tokenizers_output.json \
    --show-ids 
    --seed $((1 + $RANDOM % 10))\
    > tokenizers_output.txt

kill $NVIDIA_SMI_PID
