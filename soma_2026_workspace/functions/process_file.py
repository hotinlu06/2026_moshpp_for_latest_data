import pandas as pd
import numpy as np
import os
import glob
import ezc3d
import re
from tqdm import tqdm
from typing import Tuple, List

# ================= Helper Functions =================
def extract_marker_base_name(column_name: str) -> str:
    """Extract base Marker name (compatible with suffix removal)"""
    base_name = re.sub(r'_[XYZ]$', '', column_name, flags=re.IGNORECASE)
    base_name = re.sub(r'_Residual$', '', base_name, flags=re.IGNORECASE)
    return base_name.strip()

# ================= Core Processing Function =================
def process_csv_to_c3d(csv_path: str, output_dir: str, frame_rate: float = 30.0, 
                       original_unit: str = 'm', target_unit: str = 'mm') -> str:
    """
    Process a single CSV: Read data -> Calculate Residual in memory to filter bad points -> Unit conversion -> Export to C3D
    """
    try:
        # Unit conversion factors
        UNIT_FACTORS = {
            ('m', 'mm'): 1000.0,
            ('mm', 'm'): 0.001,
            ('m', 'm'): 1.0,
            ('mm', 'mm'): 1.0
        }
        factor = UNIT_FACTORS.get((original_unit.lower(), target_unit.lower()), 1.0)
        
        # 1. Read CSV file
        df = pd.read_csv(csv_path, engine='python')
        df.columns = [str(c).strip() for c in df.columns]
        
        if "Frame" not in df.columns:
            print(f"[Skip] {os.path.basename(csv_path)} is missing the Frame column")
            return None
            
        frames = df["Frame"].values
        num_frames = len(frames)
        
        # 2. Smart parsing to extract valid Marker XYZ column mappings
        marker_columns = {} 
        for col in df.columns:
            if col.lower() == "frame":
                continue
            if re.search(r'_[XYZ]$', col, re.IGNORECASE):
                marker_name = extract_marker_base_name(col)
                axis = col.rsplit('_', 1)[1].lower()
                if marker_name not in marker_columns:
                    marker_columns[marker_name] = {}
                marker_columns[marker_name][axis] = col
                
        valid_markers = [m for m, axes in marker_columns.items() if {'x', 'y', 'z'}.issubset(axes.keys())]
        num_markers = len(valid_markers)
        if num_markers == 0:
            print(f"[Skip] {os.path.basename(csv_path)} did not find complete XYZ Markers")
            return None
            
        # 3. Initialize C3D matrices
        c3d_points = np.zeros((4, num_markers, num_frames), dtype=np.float32)
        c3d_residuals = np.zeros((1, num_markers, num_frames), dtype=np.float32) # Exclusive residual matrix
        missing_count_in_memory = 0 
        
        # 4. Extract data, process Residual and bad point mask (🔥 Core logic fusion area)
        for marker_idx, marker in enumerate(valid_markers):
            axes = marker_columns[marker]
            
            # Extract unscaled raw values (corresponding to Code 1 logic)
            raw_x = pd.to_numeric(df[axes['x']], errors='coerce').values
            raw_y = pd.to_numeric(df[axes['y']], errors='coerce').values
            raw_z = pd.to_numeric(df[axes['z']], errors='coerce').values
            
            # 💡 Code 1 logic: Determine missing values (if there are NaNs, or XYZ coordinates are 0 simultaneously, consider as a bad point)
            bad_mask = np.isnan(raw_x) | np.isnan(raw_y) | np.isnan(raw_z) | ((raw_x == 0) & (raw_y == 0) & (raw_z == 0))
            
            # Generate Residual array: bad points set to -1.0, normal points set to 1.0
            res_vals = np.where(bad_mask, -1.0, 1.0)
            
            # Apply unit scaling (corresponding to Code 2 logic)
            x_vals = raw_x * factor
            y_vals = raw_y * factor
            z_vals = raw_z * factor
            
            # 💡 Code 2 logic linkage: Forcibly wash coordinates at bad point locations to NaN
            x_vals[bad_mask] = np.nan
            y_vals[bad_mask] = np.nan
            z_vals[bad_mask] = np.nan
            
            missing_count_in_memory += np.sum(bad_mask)
            
            # Write to point matrix (the 4th dimension must be 1.0)
            c3d_points[0, marker_idx, :] = x_vals
            c3d_points[1, marker_idx, :] = y_vals
            c3d_points[2, marker_idx, :] = z_vals
            c3d_points[3, marker_idx, :] = 1.0  
            
            # Write to residual matrix (stored directly here, reserved for ezc3d)
            c3d_residuals[0, marker_idx, :] = res_vals
            
        # 5. Write to C3D object
        c3d = ezc3d.c3d()
        
        c3d.add_parameter("POINT", "RATE", [frame_rate]) 
        c3d.add_parameter("POINT", "LABELS", valid_markers) 
        c3d.add_parameter("POINT", "USED", [num_markers]) 
        c3d.add_parameter("POINT", "UNITS", [target_unit]) 
        c3d.add_parameter("POINT", "SCALE", [-1.0]) 
        c3d.add_parameter("FRAME", "FIRST", [int(frames[0])]) 
        c3d.add_parameter("FRAME", "LAST", [int(frames[-1])]) 
        
        # Mount data: points to points, residuals to residuals
        c3d["data"]["points"] = c3d_points
        
        if "meta_points" not in c3d["data"]:
            c3d["data"]["meta_points"] = {}
        c3d["data"]["meta_points"]["residuals"] = c3d_residuals
            
        # 6. Save to disk
        os.makedirs(output_dir, exist_ok=True)
        c3d_filename = os.path.splitext(os.path.basename(csv_path))[0] + ".c3d"
        c3d_path = os.path.join(output_dir, c3d_filename)
        c3d.write(c3d_path)
        
        return c3d_path
        
    except Exception as e:
        print(f"\n[Conversion Failed] {os.path.basename(csv_path)}: {str(e)}")
        return None

# ================= Batch Processing Framework =================
def batch_convert_pipeline(csv_dir: str, output_dir: str, frame_rate: float = 30.0,
                           original_unit: str = 'm', target_unit: str = 'mm'):
    """
    Batch processing pipeline controller
    """
    if not os.path.isdir(csv_dir):
        raise ValueError(f"Input path is not a directory: {csv_dir}")
        
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in the directory: {csv_dir}")
        return
        
    print(f"[Batch Conversion] Found {len(csv_files)} CSV files, starting Residual filtering and C3D construction...")
    success_count = 0
    failed_files = []
    
    for csv_path in tqdm(csv_files, desc="Overall Processing Progress"):
        result = process_csv_to_c3d(
            csv_path=csv_path,
            output_dir=output_dir,
            frame_rate=frame_rate,
            original_unit=original_unit,
            target_unit=target_unit
        )
        if result:
            success_count += 1
        else:
            failed_files.append(os.path.basename(csv_path))
            
    print(f"\n===== Conversion Results =====")
    print(f"Total files: {len(csv_files)}")
    print(f"Successfully exported: {success_count} C3D files")
    print(f"Failed files: {len(failed_files)}")
    if failed_files:
        print("Failed list:", failed_files)

# ================= Execution Entry =================
if __name__ == "__main__":
    # Change this to your input directory (unprocessed old data)
    INPUT_DIR = "/home/hotin/2026_moshpp_for_latest_data/data_2026/04_Boss"
    
    # Change this to your output directory (one step completion)
    OUTPUT_DIR = "/home/hotin/2026_moshpp_for_latest_data/soma_2026_workspace/data/Boss_04"
    
    # Parameter settings
    FRAME_RATE = 240.0  # change this if your original data has a different frame rate
    ORIGINAL_UNIT = 'm'  
    TARGET_UNIT = 'mm'   
    
    print("🚀 Starting integrated pipeline: [CSV without Residual] -> [Auto clean/Calculate Residual] -> [Export standardized C3D]")
    batch_convert_pipeline(
        csv_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        frame_rate=FRAME_RATE,
        original_unit=ORIGINAL_UNIT,
        target_unit=TARGET_UNIT
    )