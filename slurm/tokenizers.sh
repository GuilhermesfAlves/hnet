#!/bin/bash

#SBATCH -p gpu  	           # Partition ou queue de GPU
#SBATCH --job-name=tokenizers  # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 16                  # CPUs por task
#SBATCH --gres=gpu:V100:4      # Número de GPUs (4 GPU)
#SBATCH --mem=64G              # Memória total
#SBATCH -o ./output_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./output_%j.err     # Arquivo de erro (adiciona job id %j)

srun python compare_tokenizers.py \
    --dataset-name uonlp/CulturaX \
    --config-name pt \
    --split validation \
    --json-out tokenizers_output.json \
    --show-ids \
    > tokenizers_output.txt

kill $NVIDIA_SMI_PID
