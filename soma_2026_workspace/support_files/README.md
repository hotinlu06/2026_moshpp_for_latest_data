## `support_files/` (Algorithm Dependencies & Configurations)

This directory contains all the official pre-trained human models, transformation matrices, and topology mapping files required to run the MoSh++ algorithm. To simplify deployment, all prerequisites have been pre-downloaded and configured for out-of-the-box use.

### Core Models and Parameter Files
* **`SMPL_NEUTRAL.npz` & `SMPLX_NEUTRAL.npz`**: The gender-neutral base parametric human models provided by the official source.
* **`smplx_to_smpl.pkl`**: The topology transformation matrix. It contains the core `J_regressor`, which is used after the MoSh++ solver finishes to map the high-resolution SMPL-X topology results down to the standard SMPL skeletal structure.
* **`smplx_template.obj`**: A static mesh template exported from the official model. It can be directly imported into 3D software like Blender to visually inspect the human body topology and vertex distribution.

### `conf/` (Algorithm Configurations)
Stores the underlying configuration files for MoSh++, inherited directly from the official code repository.
* **`moshpp_conf.yaml`**: The core runtime parameter configuration file. The current parameters have been adapted for this project pipeline and **do not require modification for standard use**. If you need to adjust advanced parameters like optimizer weights or iteration counts, you can review and edit this file.

### `marker_layouts/` (Marker Mapping)
Stores the spatial mapping dictionary that links motion capture markers to the surface vertices of the human model. This serves as the prior input for the MoSh++ solver.
* **Mapping Principle**: Establishes a binding relationship between physical motion capture points and the vertex indices of the SMPL-X model. For example, `"WaistLFront": 3309` indicates that the left front waist point corresponds to vertex index 3309 on the model. The current mapping baseline was derived by picking points on the `smplx_template.obj`.
* **Accuracy Calibration**: If you experience inaccurate solving for specific motions, you can import the `.obj` template into Blender, re-pick more precise vertex indices, and update this file accordingly.

> **Data Alignment Note (Important)**
>
> The preprocessed `.csv` motion capture data typically uses numerical indices as headers (e.g., `0_x`, `0_y`, `0_z`) rather than semantic labels (e.g., `WaistLFront_x`).
> Therefore, before executing the run, you **must** ensure that the keys in the mapping JSON file match the indices exported in the CSV.
> * **Modification Example**: You will need to change the original `"WaistLFront": 3309` to `"0": 3309` to ensure the algorithm correctly reads the data from the corresponding channels.