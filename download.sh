#!/bin/bash

#SBATCH -p gpu  	       # Partition ou queue de GPU
#SBATCH --job-name=download_set # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 8                   # CPUs por task
#SBATCH --mem=32G              # Memória total
#SBATCH -o ./d_%j.log     # Arquivo de log (adiciona job id %j)
#SBATCH -e ./d_%j.err     # Arquivo de erro (adiciona job id %j)
#SBATCH --time=48:00:00        # Tempo máximo de execução (hh:mm:ss)

	srun --export=ALL python download_dataset.py
