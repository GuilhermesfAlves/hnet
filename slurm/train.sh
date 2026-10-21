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

hnet_names=("hnet_2stage_L")
load_balancing_n=("6" "3,3" "6") 
lr_multipliers=("2.627,0.876" "2.627,1.517,0.715" "3.715,0.876")
#batch_sizes=(12 10 6)
batch_sizes=(10)
#sequence_lengths=(768 768 512)
sequence_lengths=(768)
max_bytes=$((5 * 1024 * 1024 * 1024)) # 5GB

for i in "${!hnet_names[@]}";do
	hnet="${hnet_names[$i]}"
	batch="${batch_sizes[$i]}"
	seq_len="${sequence_lengths[$i]}"
	load_balancing="${load_balancing_n[$i]}"
	lr_multiplier="${lr_multipliers[$i]}"
	
	echo "$hnet $batch $seq_len"
	mkdir -p gpu
	nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    		--format=csv -l 1 > gpu/gpu_memory_train_$hnet.log &
	NVIDIA_SMI_PID=$!
	export PYTORCH_ALLOC_CONF=expandable_segments:True
	srun --export=ALL python -m torch.distributed.run --nproc_per_node=4 train_pt.py \
        --dataset-name uonlp/CulturaX \
	    --dataset-config-name pt \
		--csv-path checkpoints/train_pt/$hnet.metrics.txt \
		--out-dir checkpoints/train_pt/$hnet \
	    --text-column text \
		--seq-len $seq_len \
		--batch-size $batch \
		--load-balancing-n $load_balancing \
		--lr-multiplier $lr_multiplier \
		--lr-schedule wsd \
		--warmup-frac 0.1 \
		--decay-frac 0.2 \
		--no-streaming \
		--max-tokens $max_bytes \
		--num-workers 16 \
		--save-every 1500 \
        --model-config configs/$hnet.json > output/train_output.$hnet.txt

	kill $NVIDIA_SMI_PID
done
