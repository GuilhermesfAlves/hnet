#!/bin/bash

#SBATCH -p gpu  	       # Partition ou queue de GPU
#SBATCH --job-name=hnet        # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 1                   # CPUs por task
#SBATCH --gres=gpu:1           # Número de GPUs (1 GPU)
#SBATCH --mem=8G               # Memória total
#SBATCH -o ./output_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./output_%j.err     # Arquivo de erro (adiciona job id %j)
#SBATCH --time=04:00:00        # Tempo máximo de execução (hh:mm:ss)

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > gpu_memory.log &
NVIDIA_SMI_PID=$!


echo "oi\n" | srun --export=ALL python generate.py \
	--model-path hnet_2stage_XL.pt \
	--config-path ~/hnet/configs/hnet_2stage_XL.json \
	--max-tokens 1024 \
	--temperature 1.0 \
	--top-p 1.0 > output.txt

kill $NVIDIA_SMI_PID
