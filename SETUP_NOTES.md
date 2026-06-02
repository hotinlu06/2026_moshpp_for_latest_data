# Setup fixes for the SOMA / MoSh++ pipeline

The original README was written against an older Linux box (Python 3.7,
CUDA 10.2, an old GPU). On your server (NVIDIA driver 570 / CUDA 12.8) the
install fails for a cluster of well-known reasons. This file lists what was
fixed and how to run the pipeline now.

## What was broken

| # | Symptom you would have hit | Root cause |
|---|---|---|
| 1 | `pip install torch==1.8.2+cu102` fails on Python 3.9+ / silent kernel errors at runtime on Ampere/Hopper/Ada GPUs ("no kernel image is available for execution on the device") | The `+cu102` wheel only ships kernels for compute capability ≤ 7.5. Modern GPUs need ≥ `sm_80`. |
| 2 | `conda create -n soma python=3.7` solves forever / can't find packages | Python 3.7 is EOL; conda-forge dropped most builds. |
| 3 | Import-time `AttributeError: module 'numpy' has no attribute 'bool'` (also `np.int`, `np.float`, `np.str`) | `soma/src/soma/tools/soma_processor.py:131,132,311,353…` and `chumpy` still use these. They were removed in numpy ≥ 1.24. |
| 4 | `ModuleNotFoundError: No module named 'moshpp'` when running `solve_labeled_mocap.ipynb` | `soma/src/soma/amass/mosh_manual.py:37` imports `moshpp`, but `soma/requirements.txt` doesn't list it. You have to clone `nghorbani/moshpp` separately. |
| 5 | `pip install pytorch3d` fails to build from source | Always requires the prebuilt wheel from Facebook's index that matches your `(python, cuda, torch)` triple. |
| 6 | `pip install mkl-fft mkl-random mkl-service` fails or installs broken binaries | PyPI MKL wheels are unmaintained; the moshpp/soma inference path doesn't use them — they're dropped. |
| 7 | `process_file.py` hard-codes a single scene path (`/home/u3625378/.../Boss_01`) | Replaced with CLI flags + `--recursive` to batch all 5 scenes. |

## What this repo now contains

- **`setup_env.sh`** — one-shot installer. Creates a conda env, installs a
  pinned dependency stack that resolves all of the above, clones `moshpp`,
  and runs a sanity-check import at the end.
- **`soma_2026_workspace/functions/process_file.py`** — now CLI-driven, with
  `--recursive` to convert all scenes in one call.
- **`SETUP_NOTES.md`** — this file.

The target stack is:

- Python **3.9**
- PyTorch **1.13.1 + cu117** (runs against your CUDA 12.8 host driver via
  forward compatibility, has Ampere/Hopper kernels, has matching pytorch3d
  wheels)
- numpy **1.23.5** (last release where `np.bool`/`np.int`/`np.float` still
  exist — keeps both `soma` source and `chumpy` happy)
- pytorch-lightning **1.9.5** (last 1.x; you don't run training, but soma
  imports it transitively)

If you ever upgrade torch, also bump `pytorch3d` to the wheel index that
matches the new `pyt{X}` tag in step 7 of `setup_env.sh`.

## Running on the server

```bash
# 1. install
cd ~/2026_moshpp_for_latest_data
bash setup_env.sh                       # ~10–20 min the first time
conda activate soma

# 2. CSV -> C3D for all 5 scenes
python soma_2026_workspace/functions/process_file.py \
  --input-dir  ./data_2026/Data \
  --output-dir ./soma_2026_workspace/data \
  --recursive \
  --actor Actor_01 \
  --frame-rate 240

# (this produces soma_2026_workspace/data/<scene>/Actor_01/*.c3d)

# 3. drop the prepared settings.json into each scene dir
for scene in soma_2026_workspace/data/*/Actor_01; do
  cp soma_2026_workspace/data/Actor_01/settings.json "$scene/" 2>/dev/null || true
done
# (adjust the source path above to wherever your settings.json template lives)

# 4. run MoSh++ — strongly recommended via tmux, two-stage as the README
#    describes. The notebook lives at:
#      soma/src/tutorials/solve_labeled_mocap.ipynb
#    For long runs, export it:
jupyter nbconvert --to script soma/src/tutorials/solve_labeled_mocap.ipynb
tmux new -s mosh "python soma/src/tutorials/solve_labeled_mocap.py"
```

## Verifying your GPU actually works

After `conda activate soma`, the last block of `setup_env.sh` already prints
`torch.cuda.is_available()` and the device capability. If you see something
like `cap (8, 6)` or higher, you're good. If `is_available()` is False, your
driver/runtime mismatch — bump `CUDA_TAG=cu118` at the top of the script and
re-run (cu118 is a fine choice too on a CUDA 12.8 driver).

## Things still NOT fixed in this pass

- The notebook itself still has the original author's `/home/nghorbani/...`
  paths in some `pkl` references shown in cell *outputs* — those are output
  text, not active config. Active path config you set in the first cells:
  `soma_work_base_dir`, `target_ds_names`, `parallel_cfg.max_num_jobs`.
- `body_visualizer` needs OpenGL libs on the server (`libosmesa6`, `libgl1`)
  if you want to render result GIFs. The mosh optimization itself does not.
- `recover_24_smpl.py` and `visualization_2d+3d.py` were not audited — they
  may need their own path edits, but their imports look standard.
