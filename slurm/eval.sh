#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name=hnet-eval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=64G
#SBATCH -o ./eval_%j.log
#SBATCH -e ./eval_%j.err

hnet_names=("hnet_1stage_L" "hnet_2stage_L" "hnet_1stage_Llama")
seq_lens=(768 768 512)
batch_sizes=(12 9 6)
load_balancing_n=("6" "3,3" "6")        

for i in "${!hnet_names[@]}"; do
    hnet="${hnet_names[$i]}"
    echo "Avaliando $hnet"

    srun --export=ALL python eval_pt.py \
        --model-config configs/$hnet.json \
        --checkpoint-dir checkpoints/train_pt/$hnet \
        --datasets-config eval_datasets.json \
        --csv-dir results/eval_pt/$hnet \
        --seq-len ${seq_lens[$i]} \
        --batch-size ${batch_sizes[$i]} \
        --load-balancing-n ${load_balancing_n[$i]} \
        --eval-steps 50 \
        --every 1500 \
        --include-final \
		--workers 16 \
		> output/eval_output.$hnet.txt
done