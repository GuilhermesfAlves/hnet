#!/bin/bash
#SBATCH --job-name=env_check
#SBATCH --output=env_check_%j.out
#SBATCH --error=env_check_%j.err
#SBATCH --partition=Family-vti
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

set -u

echo "========================================"
echo " SLURM / SISTEMA"
echo "========================================"

echo "Hostname:"
hostname

echo
echo "Job ID:"
echo "$SLURM_JOB_ID"

echo
echo "Node:"
echo "$SLURMD_NODENAME"

echo
echo "CUDA_VISIBLE_DEVICES:"
echo "${CUDA_VISIBLE_DEVICES:-<não definido>}"

echo
echo "========================================"
echo " PYTHON"
echo "========================================"

which python || true

python --version || true

echo
python - <<'PY'
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
PY

echo
echo "========================================"
echo " NVIDIA / DRIVER"
echo "========================================"

nvidia-smi || true

echo
echo "========================================"
echo " NVCC / CUDA TOOLKIT"
echo "========================================"

which nvcc || true
nvcc --version || true

echo
echo "CUDA_HOME:"
echo "${CUDA_HOME:-<não definido>}"

echo
echo "PATH CUDA:"
echo "$PATH" | tr ':' '\n' | grep -i cuda || true

echo
echo "LD_LIBRARY_PATH:"
echo "${LD_LIBRARY_PATH:-<não definido>}" | tr ':' '\n' | grep -i cuda || true

echo
echo "========================================"
echo " PYTORCH"
echo "========================================"

python - <<'PY'
try:
    import torch

    print("Torch version:       ", torch.__version__)
    print("Torch path:          ", torch.__file__)
    print("Torch CUDA version:  ", torch.version.cuda)
    print("CUDA available:      ", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU count:            ", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:               ", torch.cuda.get_device_name(i))
            print(f"  Compute capability: ", torch.cuda.get_device_capability(i))

    try:
        print("cuDNN version:        ", torch.backends.cudnn.version())
    except Exception as e:
        print("cuDNN version:         ERROR:", e)

except Exception as e:
    print("ERRO importando torch:")
    print(repr(e))
PY

echo
echo "========================================"
echo " PACOTES PYTHON PRINCIPAIS"
echo "========================================"

python -m pip list 2>/dev/null | grep -Ei \
    'torch|triton|flash|mamba|transformers|numpy|scipy|einops|ninja|packaging' \
    || true

echo
echo "========================================"
echo " FLASH ATTENTION"
echo "========================================"

python - <<'PY'
try:
    import flash_attn

    print("FlashAttention version:", getattr(flash_attn, "__version__", "unknown"))
    print("FlashAttention path:   ", flash_attn.__file__)

except Exception as e:
    print("ERRO importando flash_attn:")
    print(repr(e))
PY

echo
echo "========================================"
echo " TRITON"
echo "========================================"

python - <<'PY'
try:
    import triton

    print("Triton version:", triton.__version__)
    print("Triton path:   ", triton.__file__)

except Exception as e:
    print("ERRO importando triton:")
    print(repr(e))
PY

echo
echo "========================================"
echo " GCC / G++"
echo "========================================"

gcc --version | head -n 1 || true
g++ --version | head -n 1 || true

echo
echo "========================================"
echo " CONDA"
echo "========================================"

echo "CONDA_PREFIX:"
echo "${CONDA_PREFIX:-<não definido>}"

echo
echo "Conda environment:"
echo "${CONDA_DEFAULT_ENV:-<não definido>}"

echo
conda list 2>/dev/null | grep -Ei \
    'torch|pytorch|cuda|cudnn|triton|flash|numpy|ninja' \
    || true

echo
echo "========================================"
echo " VARIÁVEIS RELEVANTES"
echo "========================================"

env | grep -E \
    '^(CUDA|TORCH|TRITON|CONDA|NVIDIA|LD_LIBRARY_PATH|PATH)=' \
    | sort || true

echo
echo "========================================"
echo " FIM"
echo "========================================"
