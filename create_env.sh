#!/bin/bash

# Cria ambiente
conda create -n hnet-cuda python=3.10 -y
conda activate hnet-cuda

# Instala pytorch 2.5.1 e cuda-runtime-torch 12.1
conda install -c pytorch -c nvidia pytorch==2.5.1 pytorch-cuda=12.1 -y

# Instala o cuda-toolkit 12.8
conda install -c nvidia cuda-toolkit=12.8 -y

# Instala as dependencias do projeto
pip install -e .