#!/bin/bash

set -u

# ============================================================
# Configuração
# ============================================================

hnet_names=(
    "hnet_1stage_L"
    "hnet_1stage_XL"
    "hnet_2stage_L"
    "hnet_2stage_XL"
)

# seq_len deve ser múltiplo do chunk_size=256
sequence_lengths=(
    256
    512
    768
    1024
)

# Testaremos esses batch sizes em ordem crescente.
batch_candidates=(
    1
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

# Diretórios
METRICS_DIR="checkpoints/train_pt"
OUTPUT_DIR="output"

mkdir -p "$METRICS_DIR"
mkdir -p "$OUTPUT_DIR"

RESULTS="$OUTPUT_DIR/batch_seq_sweep.csv"

echo "model,seq_len,batch_size,tokens_per_gpu,status,max_vram_mb" \
    > "$RESULTS"


# ============================================================
# Função para obter a maior memória usada pelas GPUs
# ============================================================

get_max_vram() {
    nvidia-smi \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits \
        | awk '
        BEGIN { max=0 }
        {
            if ($1 > max)
                max=$1
        }
        END {
            print max
        }'
}


# ============================================================
# Testa uma configuração
# ============================================================

test_config() {

    local hnet="$1"
    local batch="$2"
    local seq_len="$3"

    local log_file="$OUTPUT_DIR/sweep.${hnet}.b${batch}.s${seq_len}.txt"

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
        --no-streaming \
        --model-config "configs/$hnet.json" \
        > "$log_file" 2>&1

    local exit_code=$?

    # --------------------------------------------------------
    # Verifica OOM
    # --------------------------------------------------------

    local max_vram
    max_vram=$(get_max_vram)

    if [ "$exit_code" -ne 0 ]; then

        if grep -qiE \
            "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|OOM" \
            "$log_file"; then

            echo "OOM"
            echo "$hnet,$seq_len,$batch,$((batch * seq_len)),OOM,$max_vram" \
                >> "$RESULTS"

            return 1

        else

            echo "ERRO (exit code $exit_code)"
            echo "$hnet,$seq_len,$batch,$((batch * seq_len)),ERROR,$max_vram" \
                >> "$RESULTS"

            return 2

        fi

    fi

    # --------------------------------------------------------
    # Configuração funcionou
    # --------------------------------------------------------

    echo "OK"
    echo "VRAM máxima: ${max_vram} MB"

    echo "$hnet,$seq_len,$batch,$((batch * seq_len)),OK,$max_vram" \
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

column -t -s ',' "$RESULTS"

echo
echo "Arquivo:"
echo "$RESULTS"