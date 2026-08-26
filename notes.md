# Anotações

Como as gpus do cluster são antigas, os pacotes de 3ºs foram alterados para compilar com arquitetura `sm_60` e `sm_70`. O pytorch que está sendo utilizado é o 2.10.0 com cuda 12.6 pois é o que suporta essas archs


```bash
#!/bin/bash

# Cria ambiente
conda create -n hnet2 python=3.10 -y
conda activate hnet2

# Instala pytorch 2.10.0
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

# Instala o cuda-toolkit 12.6
conda install -c nvidia cuda-toolkit=12.6 -y

# Instala as dependencias do projeto
pip install -e . --no-build-isolation --no-cache-dir --no-deps
```

---
# OLD

# Instala pytorch 2.5.1 e cuda-runtime-torch 12.1
# pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
#  --index-url https://download.pytorch.org/whl/cu121
pip install -U typing-extensions

# Instala pytorch 2.7.0 e cuda-runtime-torch 12.8
pip install torch==2.7.0  \
  --index-url https://download.pytorch.org/whl/cu128

# Instala o nvcc
sudo apt install cuda-toolkit-12-8

pip install psutil

# ================================================================================================

# Para baixar o causal-convid 1.5.0 post 8 é possível usar no máximo o torch 2.6
# Para baixar o causal-convid 1.6.0 é possível usar no máximo o torch 2.7
# cu11 significa que foi compilado com o cuda-toolkit-11 e é compativel com cuda runtime 11 pra cima
https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu11torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# ================================================================================================


# Para baixar o flash-attention 2.8.3 é necessário no máximo o torch 2.8
# cu12 significa que foi compilado com o cuda-toolkit-12 e é compativel com cuda runtime 12 pra cima
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl


# ================================================================================================


# Para baixar o mamba_ssm 2.2.5 é preciso no máximo o torch 2.7
# cu12 significa que foi compilado com o cuda-toolkit-12 e é compativel com cuda runtime 12 pra cima
https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# É possível baixar a versão 2.3.0 também
https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
