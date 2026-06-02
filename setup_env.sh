#!/usr/bin/env bash
# SOMA + MoSh++ environment setup — fixed for modern Linux servers
# (NVIDIA driver >= 525, CUDA 11/12, Python 3.7 EOL).
#
# Run from the project root: /path/to/2026_moshpp_for_latest_data
# Usage:
#   bash setup_env.sh
#
# Why the original README breaks:
#   - torch 1.8.2+cu102 has no kernels for compute capability >= 8.0
#     (Ampere/Hopper/Ada) -> "no kernel image is available for execution"
#     on your driver 570 / CUDA 12.8 host.
#   - python 3.7 is EOL; conda solver + many wheel indexes have dropped it.
#   - numpy >= 1.24 removed np.bool / np.int / np.float / np.str, which
#     soma/src/soma/tools/soma_processor.py and chumpy still use.
#   - moshpp (the actual optimizer) is *not* listed in requirements.txt
#     but soma/src/soma/amass/mosh_manual.py imports it.
#   - pytorch3d never installs from PyPI; needs Facebook's prebuilt wheel.
#   - mkl-fft/mkl-random/mkl-service from PyPI are flaky on Linux; drop them
#     (the moshpp/soma inference path does not use them).

set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$(pwd)}"
ENV_NAME="${ENV_NAME:-soma}"
PY_VER="${PY_VER:-3.9}"

# Torch / CUDA target. cu117 wheels run fine against a CUDA 12.8 host driver
# (forward compatibility), have prebuilt pytorch3d wheels, and include
# Ampere/Hopper kernels. Switch to cu118 if you prefer.
TORCH_VER="${TORCH_VER:-1.13.1}"
TV_VER="${TV_VER:-0.14.1}"
TA_VER="${TA_VER:-0.13.1}"
CUDA_TAG="${CUDA_TAG:-cu117}"

echo "[setup] project root : $PROJ_ROOT"
echo "[setup] conda env    : $ENV_NAME (python $PY_VER)"
echo "[setup] torch        : $TORCH_VER+$CUDA_TAG"

# --- 1. conda env -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not on PATH" >&2; exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] env $ENV_NAME already exists, reusing"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

# --- 2. native libs via conda-forge ----------------------------------------
# ezc3d is C++; conda-forge build is the only reliable path.
conda install -y -c conda-forge ezc3d=1.5.* boost=1.78.*

# --- 3. PyTorch (CUDA 11.7 wheel, runs on CUDA 12 driver) ------------------
pip install --upgrade "pip<24" "setuptools<66" wheel
pip install \
  "torch==${TORCH_VER}+${CUDA_TAG}" \
  "torchvision==${TV_VER}+${CUDA_TAG}" \
  "torchaudio==${TA_VER}" \
  --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# --- 4. Pinned numerical stack ---------------------------------------------
# numpy 1.23 keeps np.bool/np.int/np.float/np.str alive for soma + chumpy.
pip install \
  "numpy==1.23.5" \
  "scipy==1.10.1" \
  "pandas==1.5.3" \
  "scikit-learn==1.2.2" \
  "scikit-image==0.19.3" \
  "opencv-python==4.7.0.72" \
  "pillow==9.5.0" \
  "imageio==2.31.1" \
  "matplotlib==3.7.2" \
  "seaborn==0.12.2" \
  "tables==3.8.0" \
  "transforms3d==0.4.1" \
  "trimesh==3.22.4" \
  "colour==0.1.5" \
  "xlsxwriter==3.1.2" \
  "omegaconf==2.3.0" \
  "loguru==0.7.0" \
  "tqdm==4.65.0" \
  "toolz==0.12.0" \
  "six==1.16.0" \
  "notifiers==1.3.3" \
  "markdown==3.4.4" \
  "pycodestyle==2.11.0" \
  "threadpoolctl==3.2.0" \
  "tensorboard==2.11.2" \
  "pytorch-lightning==1.9.5" \
  "jupyterlab==4.0.5" \
  "ipython==8.14.0" \
  "pyOpenSSL==23.2.0"

# --- 5. chumpy: install a numpy>=1.20-tolerant fork ------------------------
# Upstream chumpy still imports np.bool/np.int at module level. mattloper's
# main branch is fine on numpy 1.23 because the names still exist; if a
# future numpy bump kills them again, switch to a maintained fork.
pip install "chumpy==0.70"

# --- 6. SMPL body family ---------------------------------------------------
pip install "smplx[all]==0.1.28"

# --- 7. pytorch3d (prebuilt FB wheel matching torch+cuda+python) -----------
PYTAG="py$(python -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')"
PT3D_URL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PYTAG}_${CUDA_TAG}_pyt1131/download.html"
pip install --no-index --no-cache-dir pytorch3d -f "$PT3D_URL" || {
  echo "[setup] WARN: pytorch3d wheel index missing for ${PYTAG}_${CUDA_TAG}_pyt1131."
  echo "         The mosh inference path does not strictly need pytorch3d;"
  echo "         it is only used by body_visualizer renders. Continuing."
}

# --- 8. nghorbani support repos --------------------------------------------
pip install \
  "git+https://github.com/nghorbani/configer.git" \
  "git+https://github.com/nghorbani/human_body_prior.git@SOMA" \
  "git+https://github.com/nghorbani/body_visualizer"

# --- 9. moshpp (MISSING from soma/requirements.txt — required by notebook) -
MOSHPP_DIR="$PROJ_ROOT/moshpp"
if [ ! -d "$MOSHPP_DIR" ]; then
  git clone https://github.com/nghorbani/moshpp.git "$MOSHPP_DIR"
fi
pip install -e "$MOSHPP_DIR"

# --- 10. soma itself --------------------------------------------------------
cd "$PROJ_ROOT/soma"
# soma/requirements.txt re-pulls everything; we've already pinned, so install
# the package only.
python setup.py develop
cd "$PROJ_ROOT"

# --- 11. sanity import ------------------------------------------------------
python - <<'PY'
import numpy, torch, chumpy, ezc3d
print("numpy   :", numpy.__version__)
print("torch   :", torch.__version__, "cuda?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device  :", torch.cuda.get_device_name(0),
          "cap", torch.cuda.get_device_capability(0))
print("chumpy  :", chumpy.__version__)
print("ezc3d   :", ezc3d.__version__ if hasattr(ezc3d,'__version__') else 'ok')
import soma, moshpp
from soma.amass.mosh_manual import mosh_manual  # noqa
print("soma + moshpp import OK")
PY

echo "[setup] DONE. Activate with: conda activate $ENV_NAME"
