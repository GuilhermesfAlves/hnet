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

# IMPORTANTE: como agora o paralelismo é feito manualmente via
# multiprocessing (16 processos Python), mantenha OMP_NUM_THREADS=1
# para evitar que bibliotecas internas (numpy, arrow, etc.) também
# tentem paralelizar e disputem os mesmos núcleos.
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false   # evita conflito com fork + threads

# ============================================================
# Rodar script otimizado (paralelo, 16 workers)
# ============================================================

srun --export=ALL python scripts/stats_dataset.py