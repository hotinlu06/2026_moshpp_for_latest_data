# MoSh++ Data Optimization Processing Pipeline (Based on SOMA)
<div align="center">
  <img src="./assets/01_boss_C_bending-down_row9_rep1_3d_front.gif" width="30%" alt="Front View" />
  <img src="./assets/01_boss_C_bending-down_row9_rep1_3d_side.gif" width="30%" alt="Side View" />
  <img src="./assets/01_boss_C_bending-down_row9_rep1_3d_three_quarter.gif" width="30%" alt="Three Quarter View" />
  <p><b>MoSh++ 3D Reconstruction Results (Front, Side, Three-Quarter View)</b></p>
</div>

This pipeline is based on the [SOMA](https://github.com/nghorbani/soma) algorithm library. It utilizes the MoSh++ algorithm to optimize the overall marker positions of motion capture data, effectively reducing issues like marker drifting and jittering.

## 0. Environment Setup and Dependency Installation

We directly use the MoSh++ calling pipeline encapsulated in the SOMA repository. The `soma` subdirectory in this repo is a patched fork of [SOMA](https://github.com/nghorbani/soma) with dependency fixes for Python 3.7 compatibility (see note below).

#### **Recommended: One-command install**
From the top-level working directory (e.g., `~/2026_moshpp_for_latest_data`):
```bash
bash install_env.sh
```
This script handles all compatibility issues automatically.

#### **Manual install (step by step)**

**Create Virtual Environment**
```bash
conda create -n soma python=3.7
conda activate soma
```

**Install ezc3d**
```bash
conda install -c conda-forge ezc3d
```

**Install PyTorch 1.8.2 (CUDA 10.2)**
```bash
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 \
  -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

**Install body_visualizer with Python 3.7 patch**

> **Note:** The latest `body_visualizer` requires Python ≥ 3.11 and torch ≥ 2.5, which is incompatible with the SOMA environment. Install it manually with the version constraints stripped:
```bash
git clone https://github.com/nghorbani/body_visualizer /tmp/body_visualizer
cd /tmp/body_visualizer
sed -i 's/requires-python = .*/requires-python = ">=3.7"/' pyproject.toml
sed -i '/torch/d' pyproject.toml
pip install --no-deps .
cd -
```

**Install remaining dependencies and SOMA**
```bash
cd soma
pip install -r requirements.txt
python setup.py develop
cd ..
```

**Restore correct PyTorch version** (some deps may upgrade it)
```bash
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 \
  -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

**Install pyrender**
```bash
pip install pyrender
```

**Verify**
```bash
python -c "import torch, soma, body_visualizer, human_body_prior; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 1. Directory Structure Overview
To ensure the code runs smoothly, this project strictly follows the directory structure from the official SOMA tutorials. The core structure is as follows:
```bash
2026_moshpp_for_latest_data/
├── soma/                      # (Original SOMA repository cloned via Git)
│   └── src/tutorials/         # Core runtime scripts location (focus on solve_labeled_mocap.ipynb)
│   └── ...
├── data_2026/                 # Stores the original segmented motion CSV files
└── soma_2026_workspace/       # Our main workspace
    ├── data/                  # Stores the converted .c3d files and configurations
    ├── functions/             # Stores preprocessing scripts (e.g., process_file.py)
    ├── runtime_mosh/          # Output files generated during runtime
    ├── support_files/         # Support files (models, mapping matrices, etc.)
    └── ...
```
#### **Note**: Each subfolder contains an independent ```README.md```. For more specific details about the internal structure, please refer to the instructions in the corresponding directories.

## 2. Data Preprocessing (CSV to C3D)
The SOMA framework requires input data to be in the ```.c3d``` format, so the labeled CSV files must be preprocessed.

**Enter the script directory:**
```bash
cd /2026_moshpp_for_latest_data/soma_2026_workspace/functions
```
Open and modify the path configurations in ```process_file.py```:

```INPUT_DIR```: Points to the directory containing the segmented CSV files (e.g., ```01_boss```).

```OUTPUT_DIR```: The destination path for the converted ```.c3d``` files. This path must correspond to the configuration in the subsequent Jupyter Notebook. (Suggested path format: ```.../soma_2026_workspace/data/Boss_test/Actor_01```).

**Run the conversion script:**
```bash
python process_file.py
```
Upon successful execution, the corresponding ```.c3d``` files will be generated in the target folder.

## 3. Scene Parameter Configuration (```settings.json````)
A ```settings.json``` configuration file must be included in every final data directory storing ```.c3d``` files (e.g., ```Actor_01```).

**Acquisition:** A pre-written general ```settings.json``` template is already provided in the ```Actor_01/``` directory.

**Batch Deployment:** If you have multiple scenes (e.g., ```01_boss``` to ```15_boss```), you need to copy this settings.json into the 15 different scene folders.

**Parameter Customization:** You can modify the parameters within the file based on actual capture conditions, such as changing the ```rotation_order``` to zyx, or adjusting the ```marker rate``` according to the actual frame rate.

## 4. Running the MoSh++ Optimization Algorithm
Once preprocessing and configuration are complete, enter the SOMA tutorial directory to run the optimization script:

```bash
cd /home/u3625378/soma_2026/src/tutorials/
```
Open ```solve_labeled_mocap.ipynb``` to make the following modifications and run it (you can refer to the example code in directory ```/2026_moshpp_for_latest_data/```):

### 4.1 Path and Target Dataset Settings
**Workspace Base Directory:** Ensure soma_work_base_dir points to ```2026_moshpp_for_latest_data/soma_2026_workspace```.

**Configure Target Folders**: Modify the target_ds_names list to include the folder names to be processed (e.g., ```target_ds_names=['Boss_01', 'Boss_02']```).

**Recommendation**: Run data for the same actor (e.g., with the suffix ```01```) in the same batch, so that the personal body shape parameters (Beta) calculated in ```Stage I``` remain consistent.

**Concurrency Count**: Modify ```max_num_jobs``` in ```parallel_cfg``` to control the number of files processed in parallel simultaneously (e.g., ```max_num_jobs=100```).

### 4.2 ```Stage I``` and ```Stage II``` Execution Logic
Since the algorithm is divided into two stages, the execution logic of the Jupyter Notebook is as follows. Please execute as needed:

**First Run (```Stage I``` - Shape Optimization):**
The first time you run the core code block, the system primarily executes ```Stage I``` (outputting ```[name]_stagei.pkl```) to optimize the actor's body shape parameters (Beta coefficients). At this point, the system typically only concurrently runs ```Stage II``` for the first file in the list (outputting ```[name]_stageii.pkl```, which contains pose parameters like rotation).

**Second Run (```Stage II``` - Batch Pose Optimization):**
To complete pose optimization for all files, it is recommended to directly copy and add a third code block identical to the second one in the Notebook.
When running this third code block, the system will detect that ```Stage I``` for the actor is already complete and will skip directly to ```Stage II```. At this point, all files pointed to by ```target_ds_names``` will undergo batch calculation for ```Stage II```, as long as it does not exceed the ```max_num_jobs``` limit.


## 5.  Complete MoSh++ Working Pipeline

To ensure the entire algorithm runs smoothly, please strictly follow these steps in order:

### Step 1: Data Preprocessing
Run the preprocessing script to convert the raw motion capture data into the C3D format required by MoSh++ and add confidence residuals.
* **Script Path:**
`.../soma_2026_workspace/functions/process_file.py`


### Step 2: Configuration Setup

1. **Input Data Configuration (`settings.json`):**
Place the `settings.json` configuration file in the scene folder where your raw data is located.
* *Example Path:* `.../data/Boss_test/Actor_01/settings.json`


2. **Marker Mapping Configuration (`<name>_smplx.json`):**
Create a JSON configuration file with the same name as the scene in the MoSh++ output directory.
* *Example Path:* `.../runtime_mosh/mosh_results/Boss_01/Boss_01_smplx.json`

### Step 3: Execution (Run MoSh++)

Navigate to your cloned `soma` repository and find the core computation file:

* **File Path:** `soma/src/tutorials/solve_labeled_mocap.ipynb`
* **Instructions:**
1. Open the file and modify the **input/output paths** and **folder names** according to your current project (refer to previous instructions for specific modifications).
2. **Highly Recommended:** Because the algorithm takes a very long time to run, it is strongly recommended that you convert it into a `.py` script and run it in the background using `tmux` to prevent the task from terminating due to a network disconnection.

### Step 4: Post-processing and Dimensionality Reduction

After the `Stage I` and `Stage II` computations are complete, perform dimensionality reduction on the generated complex high-dimensional parameters to restore them to standard 24-joint 3D coordinates.

* **Execution Script:**
```bash
2026_moshpp_for_latest_data/soma_2026_workspace/functions/recover_24_smpl.py

```


* *Output:* Generates the final Moshized data (`.pkl` format).

### Step 5: Visualization and Validation

Verify the fit and physical plausibility of the final 3D coordinates from a dual perspective (2D projection and 3D space).

* **Execution Script:**
```bash
python .../soma_2026_workspace/functions/visualization_2d+3d.py
```
