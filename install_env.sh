#!/bin/bash
# Environment setup script for 2026_moshpp_for_latest_data
# Handles Python 3.7 / PyTorch 1.8 compatibility issues with newer dependency versions.

set -e

echo "=== Step 1: Create conda environment ==="
conda create -n soma python=3.7 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate soma

echo "=== Step 2: Install ezc3d ==="
conda install -c conda-forge ezc3d -y

echo "=== Step 3: Install PyTorch 1.8.2 (CUDA 10.2) ==="
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 \
  -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

echo "=== Step 4: Install body_visualizer (patch for Python 3.7 compatibility) ==="
# The latest body_visualizer requires Python>=3.11 and torch>=2.5.
# We clone, strip those constraints, and install with --no-deps.
TMP_BV=$(mktemp -d)
git clone https://github.com/nghorbani/body_visualizer "$TMP_BV/body_visualizer"
cd "$TMP_BV/body_visualizer"
sed -i 's/requires-python = .*/requires-python = ">=3.7"/' pyproject.toml
sed -i '/torch/d' pyproject.toml
pip install --no-deps .
cd -
rm -rf "$TMP_BV"

echo "=== Step 5: Install remaining dependencies ==="
cd soma
pip install -r requirements.txt
python setup.py develop
cd ..

echo "=== Step 6: Restore correct PyTorch version ==="
# Some dependencies may have upgraded torch; pin it back.
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 \
  -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

echo "=== Step 7: Install pyrender ==="
pip install pyrender

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
import soma
import body_visualizer
import human_body_prior
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('All imports OK')
"
