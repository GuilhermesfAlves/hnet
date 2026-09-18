#!/bin/bash

#SBATCH -p gpu  	       # Partition ou queue de GPU
#SBATCH --job-name=hnet-train  # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 16                  # CPUs por task
#SBATCH --gres=gpu:V100:4      # Número de GPUs (4 GPU)
#SBATCH --mem=64G              # Memória total
#SBATCH -o ./output_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./output_%j.err     # Arquivo de erro (adiciona job id %j)

hnet_names=("hnet_1stage_Llama")
batch_sizes=(8)
sequence_lengths=(512)

for i in "${!hnet_names[@]}";do
	hnet="${hnet_names[$i]}"
	batch="${batch_sizes[$i]}"
	seq_len="${sequence_lengths[$i]}"

	echo "$hnet $batch $seq_len"
	mkdir -p gpu
	nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    		--format=csv -l 1 > gpu/gpu_memory_train_$hnet.log &
	NVIDIA_SMI_PID=$!
	srun --export=ALL python -m torch.distributed.run --nproc_per_node=4 train_pt.py \
		--dataset-name uonlp/CulturaX \
		--dataset-config-name pt \
		--csv-path checkpoints/train_pt/$hnet.metrics.txt \
		--out-dir checkpoints/train_pt/$hnet \
		--text-column text \
		--seq-len $seq_len \
		--batch-size $batch \
		--no-streaming \
		--max-tokens 500_000_000 \
		--num-workers 16 \
		--resume-from checkpoints/train_pt/$hnet/step_9000.pt \
		--path-output-boundary-probe texts/output/$hnet \
		--eval-every 1_000 \
		--model-config configs/$hnet.json > output/train_output.$hnet.txt

	kill $NVIDIA_SMI_PID
done
