#!/bin/bash

echo "===== SETUP START ====="

# create conda environment
conda create -n gnr_project_env python=3.11 -y

# Try activating the conda environment
# Conda initialize might be needed for bash scripts
source $(conda info --base)/etc/profile.d/conda.sh || true
conda activate gnr_project_env

# upgrade pip
pip install --upgrade pip

# install dependencies
pip install -r requirements.txt

echo "===== DOWNLOADING MODELS ====="
# Download Qwen2-VL-7B-Instruct locally so no internet is needed later
python -c "
import os
from huggingface_hub import snapshot_download

# Create models directory
os.makedirs('models', exist_ok=True)

# Download the model (safetensors only to save download time and disk space)
print('Downloading Qwen/Qwen2-VL-7B-Instruct...')
snapshot_download(
    repo_id='Qwen/Qwen2-VL-7B-Instruct',
    local_dir='./models/Qwen2-VL-7B-Instruct',
    ignore_patterns=['*.pt', '*.bin']
)
print('Download complete!')
"

echo "===== SETUP COMPLETE ====="
