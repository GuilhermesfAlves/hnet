#!/bin/bash

#SBATCH -p gpu  	       # Partition ou queue de GPU
#SBATCH --job-name=hnet-train  # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 8                   # CPUs por task
#SBATCH --gres=gpu:V100:4      # Número de GPUs (2 GPU)
#SBATCH --mem=32G              # Memória total
#SBATCH -o ./output_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./output_%j.err     # Arquivo de erro (adiciona job id %j)
#SBATCH --time=48:00:00        # Tempo máximo de execução (hh:mm:ss)

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > gpu_memory_train.log &
NVIDIA_SMI_PID=$!

hnet_names=("hnet_1stage_L" "hnet_1stage_XL" "hnet_2stage_L" "hnet_2stage_XL")
batch_sizes=(12 10 8 8)
sequence_lengths=(768 768 768 768)

for i in "${!hnet_names[@]}";do
	hnet="${hnet_names[$i]}"
	batch="${batch_sizes[$i]}"
	seq_len="${sequence_lengths[$i]}"

	echo "$hnet $batch $seq_len"
	
	srun --export=ALL python -m torch.distributed.run --nproc_per_node=4 train_pt.py \
        	--dataset-name uonlp/CulturaX \
	        --dataset-config-name pt \
			--csv-path checkpoints/train_pt/$hnet.metrics.txt \
	        --text-column text \
			--seq-len $seq_len \
			--batch-size $batch \
			--no-streaming \
        	--model-config configs/$hnet.json > output/train_output.$hnet.txt
done
kill $NVIDIA_SMI_PID
