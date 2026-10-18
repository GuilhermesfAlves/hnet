#!/bin/bash

#SBATCH -p gpu  	       # Partition ou queue de GPU
#SBATCH --job-name=hnet-tokenizers  # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 1                   # CPUs por task
#SBATCH --gres=gpu:V100:1      # Número de GPUs (1 GPU)
#SBATCH --mem=40G              # Memória total
#SBATCH -o ./tkn_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./tkn_%j.err     # Arquivo de erro (adiciona job id %j)

srun python compare_tokenizers_hnet.py \
    --dataset-name uonlp/CulturaX \
    --tokenizers bpe,tiktoken,wordpiece,sentencepiece,spacy,hnet_1stage_L,hnet_2stage_L,hnet_2stage_L_s0,hnet_1stage_Llama \
    --config-name pt \
    --hnet-1stage-L-config configs/hnet_1stage_L.json \
    --hnet-1stage-L-ckpt checkpoints/train_pt/hnet_1stage_L/final.pt \
    --hnet-2stage-L-config configs/hnet_2stage_L.json \
    --hnet-2stage-L-ckpt checkpoints/train_pt/hnet_2stage_L/final.pt \
    --hnet-1stage-Llama-config configs/hnet_1stage_Llama.json \
    --hnet-1stage-Llama-ckpt checkpoints/train_pt/hnet_1stage_Llama/final.pt \
    --split "train" \
    --json-out tokenizers_output.json \
    --show-ids \
    --seed $((1 + $RANDOM % 10))\
    > tokenizers_output.txt

kill $NVIDIA_SMI_PID
