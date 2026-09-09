#!/bin/bash

#SBATCH -p gpu  	       # Partition ou queue de GPU
#SBATCH --job-name=hnet        # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 1                   # CPUs por task
#SBATCH --gres=gpu:1           # Número de GPUs (1 GPU)
#SBATCH --mem=16G              # Memória total
#SBATCH -o ./output_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./output_%j.err     # Arquivo de erro (adiciona job id %j)
#SBATCH --time=04:00:00        # Tempo máximo de execução (hh:mm:ss)

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > gpu_memory.log &
NVIDIA_SMI_PID=$!

text=$'Artificial intelligence frameworks require robust architectures to process high-dimensional representations efficiently.\nModern neural networks bridge the gap between statistical pattern recognition and abstract semantic understanding.\nResearchers continually evaluate whether scaling parameters inherently leads to emergent cognitive capabilities or merely optimizes interpolation across vast datasets.\nPerformance benchmarks dictate the evolution of multi-stage networks in contemporary machine learning pipelines.'

hnet_configs=("hnet_1stage_L" "hnet_1stage_XL" "hnet_2stage_L" "hnet_2stage_XL")

for hnet in "${hnet_configs[@]}";do
	echo $hnet
	printf '%s\n' "$text" | srun --export=ALL python generate.py \
		--model-path $hnet.pt \
		--config-path ~/hnet/configs/$hnet.json \
		--max-tokens 1024 \
		--temperature 1.0 \
		--top-p 1.0 > output.$hnet.txt
done
kill $NVIDIA_SMI_PID
