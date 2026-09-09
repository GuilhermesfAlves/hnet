#!/bin/bash
 
#SBATCH -p gpu
#SBATCH --job-name=stats_dataset
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -o ./stats_%j.log
#SBATCH -e ./stats_%j.err
#SBATCH --time=72:00:00         
 
# ============================================================
# Variáveis de otimização
# ============================================================
 
export OMP_NUM_THREADS=1        # Evita oversubscription
export TOKENIZERS_PARALLELISM=true
 
# ============================================================
# Rodar script otimizado
# ============================================================
 
srun --export=ALL python scripts/stats_dataset_optimized.py
 