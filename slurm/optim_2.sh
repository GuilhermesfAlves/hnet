#!/bin/bash

#SBATCH -p gpu  	       # Partition ou queue de GPU
#SBATCH --job-name=hnet-optim  # Nome do job
#SBATCH -N 1                   # Número de nós (1 nó)
#SBATCH -n 1                   # Número de tasks
#SBATCH -c 16                  # CPUs por task
#SBATCH --gres=gpu:V100:4      # Número de GPUs (2 GPU)
#SBATCH -o ./optim_%j.log      # Arquivo de log (adiciona job id %j)
#SBATCH -e ./optim_%j.err      # Arquivo de erro (adiciona job id %j)

set -u

# ============================================================
# Configuração
# ============================================================

hnet_names=(
    "hnet_1stage_L"
    "hnet_1stage_Llama"
    "hnet_2stage_L"
)

# seq_len deve ser múltiplo do chunk_size=256
sequence_lengths=(
    512
    768
    1024
)

# Testaremos esses batch sizes em ordem crescente.
batch_candidates=(
    2
    4
    6
    8
    10
    12
    16
)

# Não queremos chegar exatamente aos 32 GB.
# 30 GB = margem de ~2 GB para CUDA/NCCL/kernels.
MAX_VRAM_MB=30720

# Quantos steps executar no teste.
#
# IMPORTANTE:
# Adapte essa variável para o argumento que seu train_pt.py
# usa para limitar steps, caso exista.
BENCHMARK_STEPS=20

# Intervalo (segundos) de amostragem do nvidia-smi durante cada teste.
NVIDIA_SMI_INTERVAL=1

# Diretórios
METRICS_DIR="checkpoints/train_pt/optim2"
OUTPUT_DIR="output/optim2"
SMI_DIR="$OUTPUT_DIR/nvidia_smi_logs"

mkdir -p "$METRICS_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SMI_DIR"

RESULTS="$OUTPUT_DIR/batch_seq_sweep.csv"

echo "model,seq_len,batch_size,tokens_per_gpu,status,max_vram_mb,nvidia_smi_max_vram_mb,nvidia_smi_max_vram_gpu" \
    > "$RESULTS"

# ------------------------------------------------------------
# PID do nvidia-smi em background (rastreado globalmente para o
# trap de saída conseguir limpar mesmo se o script for
# interrompido no meio de um teste).
# ------------------------------------------------------------

SMI_PID=""

cleanup_smi() {
    if [ -n "$SMI_PID" ] && kill -0 "$SMI_PID" 2>/dev/null; then
        kill "$SMI_PID" 2>/dev/null
        wait "$SMI_PID" 2>/dev/null
    fi
    SMI_PID=""
}

# Garante que nenhum nvidia-smi fique "pendurado" se o job for
# cancelado (timeout do Slurm, Ctrl+C, etc.)
trap cleanup_smi EXIT INT TERM

# ============================================================
# Testa uma configuração
# ============================================================

test_config() {

    local hnet="$1"
    local batch="$2"
    local seq_len="$3"

    local log_file="$OUTPUT_DIR/sweep.${hnet}.b${batch}.s${seq_len}.txt"
    local smi_log="$SMI_DIR/sweep.${hnet}.b${batch}.s${seq_len}.csv"

    echo
    echo "============================================================"
    echo "Modelo : $hnet"
    echo "Batch  : $batch"
    echo "Seq    : $seq_len"
    echo "Tokens : $((batch * seq_len)) / GPU"
    echo "============================================================"

    # --------------------------------------------------------
    # Limpa cache de memória CUDA de processos anteriores
    # --------------------------------------------------------

    sync

    # --------------------------------------------------------
    # Inicia o nvidia-smi em background, amostrando todas as
    # GPUs do nó a cada NVIDIA_SMI_INTERVAL segundos, durante
    # toda a execução do teste.
    # --------------------------------------------------------

    nvidia-smi \
        --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
        --format=csv \
        -l "$NVIDIA_SMI_INTERVAL" \
        > "$smi_log" 2>/dev/null &
    SMI_PID=$!

    srun --export=ALL \
        python -m torch.distributed.run \
        --nproc_per_node=4 \
        train_pt.py \
        --dataset-name uonlp/CulturaX \
        --dataset-config-name pt \
        --csv-path "$METRICS_DIR/sweep_${hnet}_b${batch}_s${seq_len}.metrics.txt" \
        --text-column text \
        --seq-len "$seq_len" \
        --batch-size "$batch" \
        --max-steps "$BENCHMARK_STEPS" \
	--num-workers 16 \
        --benchmark \
        --no-streaming \
        --model-config "configs/$hnet.json" \
        > "$log_file" 2>&1

    local exit_code=$?

    # --------------------------------------------------------
    # Para o nvidia-smi assim que o treino termina — sem isso
    # ele continuaria rodando indefinidamente (-l é um loop).
    # --------------------------------------------------------

    cleanup_smi

    # --------------------------------------------------------
    # VRAM segundo o próprio PyTorch (torch.cuda.max_memory_allocated)
    # --------------------------------------------------------

    local max_vram
    max_vram=$(grep 'BENCHMARK_VRAM_ALLOCATED_MB=' "$log_file" |
        sed 's/.*=//' |
        sort -n |
        tail -1)

    # --------------------------------------------------------
    # VRAM segundo o nvidia-smi (uso real da placa, incluindo
    # overhead de CUDA context/NCCL que o PyTorch não reporta).
    # Pega o maior valor de memory.used entre TODAS as GPUs e
    # TODOS os timestamps amostrados durante o teste, e também
    # qual GPU (index) teve esse pico.
    # --------------------------------------------------------

    local nvidia_smi_result
    nvidia_smi_result=$(awk -F',' '
        NR == 1 { next }  # pula o header do nvidia-smi
        {
            gsub(/ MiB/, "", $4)
            gsub(/^[ \t]+|[ \t]+$/, "", $4)
            gsub(/^[ \t]+|[ \t]+$/, "", $2)
            mem = $4 + 0
            if (mem > max) {
                max = mem
                max_gpu = $2
            }
        }
        END {
            if (max == "") { print "0,NA" }
            else { print max "," max_gpu }
        }
    ' "$smi_log")

    local nvidia_smi_max_vram="${nvidia_smi_result%%,*}"
    local nvidia_smi_max_gpu="${nvidia_smi_result##*,}"

    # --------------------------------------------------------
    # Verifica OOM
    # --------------------------------------------------------

    if [ "$exit_code" -ne 0 ]; then

        if grep -qiE \
            "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|OOM" \
            "$log_file"; then

            echo "OOM"
            echo "VRAM (torch): ${max_vram} MB | VRAM (nvidia-smi): ${nvidia_smi_max_vram} MB (GPU $nvidia_smi_max_gpu)"
            echo "$hnet,$seq_len,$batch,$((batch * seq_len)),OOM,$max_vram,$nvidia_smi_max_vram,$nvidia_smi_max_gpu" \
                >> "$RESULTS"

            return 1

        else

            echo "ERRO (exit code $exit_code)"
            echo "$hnet,$seq_len,$batch,$((batch * seq_len)),ERROR,$max_vram,$nvidia_smi_max_vram,$nvidia_smi_max_gpu" \
                >> "$RESULTS"

            return 2

        fi

    fi

    # --------------------------------------------------------
    # Configuração funcionou
    # --------------------------------------------------------

    echo "OK"
    echo "VRAM (torch): ${max_vram} MB | VRAM (nvidia-smi): ${nvidia_smi_max_vram} MB (GPU $nvidia_smi_max_gpu)"
    echo "$hnet,$seq_len,$batch,$((batch * seq_len)),OK,$max_vram,$nvidia_smi_max_vram,$nvidia_smi_max_gpu" \
        >> "$RESULTS"
    return 0
}


# ============================================================
# Sweep
# ============================================================

for hnet in "${hnet_names[@]}"; do
    echo
    echo
    echo "############################################################"
    echo "MODELO: $hnet"
    echo "############################################################"

    # --------------------------------------------------------
    # Para cada seq_len
    # --------------------------------------------------------

    for seq_len in "${sequence_lengths[@]}"; do

        echo
        echo "------------------------------------------------------------"
        echo "$hnet - seq_len=$seq_len"
        echo "------------------------------------------------------------"

        # Maior batch que funcionou nesse seq_len
        last_good_batch=0

        for batch in "${batch_candidates[@]}"; do

            test_config "$hnet" "$batch" "$seq_len"

            status=$?

            # ------------------------------------------------
            # Funcionou
            # ------------------------------------------------

            if [ "$status" -eq 0 ]; then
                last_good_batch="$batch"
                continue
            fi

            # ------------------------------------------------
            # OOM:
            # não precisamos testar batches maiores
            # ------------------------------------------------

            if [ "$status" -eq 1 ]; then
                echo
                echo "OOM em batch=$batch."
                echo "Maior batch que funcionou: $last_good_batch"
                break
            fi

            # ------------------------------------------------
            # Outro erro:
            # interrompe esse seq_len
            # ------------------------------------------------

            if [ "$status" -eq 2 ]; then
                echo "Erro inesperado."
                break
            fi

        done
    done
done

# ============================================================
# Resultado
# ============================================================
echo
echo
echo "############################################################"
echo "RESULTADOS"
echo "############################################################"

awk -F',' '{
    for (i = 1; i <= NF; i++)
        printf "%-20s", $i
    print ""
}' "$RESULTS"

echo
echo "Arquivo de resultados:"
echo "$RESULTS"
echo
echo "Logs de nvidia-smi por configuração em:"
echo "$SMI_DIR/"
