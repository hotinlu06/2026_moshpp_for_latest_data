import pickle
import numpy as np
import torch
import smplx
import os
import logging
import glob
import shutil
import scipy.sparse as sp

# ================= Configure Logging =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Perfect_SMPL24_Builder')

# ================= Core Processing Class =================
class PerfectSMPL24Reconstructor:
    def __init__(self, smplx_model_path, smpl_model_path, transfer_pkl_path, device):
        self.device = device
        
        # 1. Load SMPL-X model (used to generate high-precision vertices)
        logger.info("📦 Loading SMPL-X model...")
        self.smplx_model = smplx.create(
            model_path=os.path.dirname(smplx_model_path),
            model_type='smplx',
            gender='neutral',
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
            ext='npz'
        ).to(self.device)
        
        # 2. Load standard SMPL model (only to borrow its J_regressor)
        # ⚠️ Note: Make sure smpl_model_path points to the directory containing SMPL_NEUTRAL.npz (or pkl)
        logger.info("📦 Loading J_regressor from standard SMPL model...")
        self.smpl_model = smplx.create(
            model_path=os.path.dirname(smpl_model_path),
            model_type='smpl',
            gender='neutral',
            batch_size=1,
            ext='npz' # Change to 'pkl' if your standard model is .pkl
        ).to(self.device)
        self.smpl_J_regressor = self.smpl_model.J_regressor # Shape: (24, 6890)
        
        # 3. Load official vertex mapping matrix (SMPL-X 10475 -> SMPL 6890)
        logger.info("📦 Loading official vertex mapping matrix (smplx_to_smpl.pkl)...")
        with open(transfer_pkl_path, 'rb') as f:
            try:
                transfer_data = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                transfer_data = pickle.load(f, encoding='latin1')
                
        matrix = transfer_data['matrix']
        if sp.issparse(matrix):
            matrix = matrix.toarray()
            
        # Convert to PyTorch Tensor, Shape: (6890, 10475)
        self.transfer_matrix = torch.tensor(matrix, dtype=torch.float32, device=self.device)
        logger.info("✅ All tools loaded successfully, ready for spatial dimensionality reduction conversion!")

    def process_single_pkl(self, stageii_path, output_dir):
        logger.info(f"Processing: {os.path.basename(stageii_path)}")
        with open(stageii_path, 'rb') as f:
            data = pickle.load(f)
        
        trans = data['trans']
        num_frames = trans.shape[0]
        raw_betas = data['betas']
        
        # Prepare Betas
        betas_padded = np.zeros(self.smplx_model.num_betas, dtype=np.float32)
        actual_len = min(len(raw_betas), self.smplx_model.num_betas)
        betas_padded[:actual_len] = raw_betas[:actual_len]
        betas_tensor = torch.tensor(betas_padded, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Prepare Fullpose / Body Pose
        if 'fullpose' in data:
            fullpose = data['fullpose']
            global_orient = fullpose[:, :3]
            body_pose = fullpose[:, 3:66] 
        elif 'body_pose' in data:
            global_orient = data.get('global_orient', np.zeros((num_frames, 3)))
            body_pose = data['body_pose']
        else:
            raise ValueError(f"❌ Missing pose parameters!")

        model_bp_dim = self.smplx_model.body_pose.shape[-1]
        if body_pose.shape[-1] < model_bp_dim:
            bp_padded = np.zeros((num_frames, model_bp_dim), dtype=np.float32)
            bp_padded[:, :body_pose.shape[-1]] = body_pose
            body_pose = bp_padded
        else:
            body_pose = body_pose[:, :model_bp_dim]

        global_orient_tensor = torch.tensor(global_orient, dtype=torch.float32, device=self.device)
        body_pose_tensor = torch.tensor(body_pose, dtype=torch.float32, device=self.device)
        trans_tensor = torch.tensor(trans, dtype=torch.float32, device=self.device)
        
        batch_size = 256 # Slightly reduce batch size, since we are outputting large Vertices
        all_smpl_joints = []
        
        num_expr = getattr(self.smplx_model, 'num_expression_coeffs', 10)
        num_left_hand = self.smplx_model.left_hand_pose.shape[-1]
        num_right_hand = self.smplx_model.right_hand_pose.shape[-1]
        
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            bs = end - i
            
            # Pad parameters
            zeros_3 = torch.zeros([bs, 3], dtype=torch.float32, device=self.device)
            zeros_expr = torch.zeros([bs, num_expr], dtype=torch.float32, device=self.device)
            zeros_left_hand = torch.zeros([bs, num_left_hand], dtype=torch.float32, device=self.device)
            zeros_right_hand = torch.zeros([bs, num_right_hand], dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                # 1. Drive SMPL-X, [Key: return_verts=True is required]
                output = self.smplx_model(
                    betas=betas_tensor.expand(bs, -1),
                    global_orient=global_orient_tensor[i:end],
                    body_pose=body_pose_tensor[i:end],
                    transl=trans_tensor[i:end],
                    jaw_pose=zeros_3,        
                    leye_pose=zeros_3,       
                    reye_pose=zeros_3,       
                    left_hand_pose=zeros_left_hand, 
                    right_hand_pose=zeros_right_hand,
                    expression=zeros_expr,     
                    return_verts=True  # 💡 Get 10475 vertices
                )
            
            smplx_verts = output.vertices # Shape (Batch, 10475, 3)
            
            # 💡 [Core Magic 1]: Map 10475 vertices down to 6890 vertices of standard SMPL
            # einsum calculation: (6890, 10475) x (Batch, 10475, 3) -> (Batch, 6890, 3)
            smpl_verts = torch.einsum('ij,bjk->bik', self.transfer_matrix, smplx_verts)
            
            # 💡 [Core Magic 2]: Regress the orthodox 24 joints from the perfect standard mesh!
            # einsum calculation: (24, 6890) x (Batch, 6890, 3) -> (Batch, 24, 3)
            smpl_24_joints = torch.einsum('ij,bjk->bik', self.smpl_J_regressor, smpl_verts)
            
            all_smpl_joints.append(smpl_24_joints.cpu().numpy())
        
        final_joints = np.concatenate(all_smpl_joints, axis=0) # Final shape: (Frames, 24, 3)
        
        base_name = os.path.basename(stageii_path).replace('.pkl', '')
        output_path = os.path.join(output_dir, f"{base_name}.pkl")
        
        with open(output_path, 'wb') as f:
            pickle.dump(final_joints, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"✅ Perfect conversion! Shoulders and wrists restored to standard body -> {output_path}")

# ================= Main Entry Point =================
if __name__ == "__main__":
    
    # 🛑 Please make sure to check these 4 paths
    # Change your input directory to the folder containing your Stage II .pkl files in runtime_mosh/mosh_results
    INPUT_DIR = "/home/u3625378/2026_moshpp_for_latest_data/soma_2026_workspace/runtime_mosh/mosh_results/Gallery_01/Actor_01"  #
    OUTPUT_DIR = "/home/u3625378/data/smpl_24_pkls_2026_perfect" # change to your desired output directory for the final 24-joint SMPL .pkl files
    
    SMPLX_MODEL_PATH = "/home/u3625378/2026_moshpp_for_latest_data/soma_2026_workspace/support_files/SMPLX_NEUTRAL.npz"
    # 💡 New: Standard SMPL model path (You need a standard SMPL model file to extract its joint regressor)
    SMPL_MODEL_PATH = "/home/u3625378/2026_moshpp_for_latest_data/soma_2026_workspace/support_files/SMPL_NEUTRAL.npz" 
    
    TRANSFER_PKL_PATH = "/home/u3625378/2026_moshpp_for_latest_data/soma_2026_workspace/support_files/smplx_to_smpl.pkl"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Current compute device: {device}")
    
    try:
        reconstructor = PerfectSMPL24Reconstructor(
            SMPLX_MODEL_PATH, 
            SMPL_MODEL_PATH, 
            TRANSFER_PKL_PATH, 
            device
        )
    except Exception as e:
        logger.error(f"❌ Initialization failed, please check if model file paths are correct: {e}")
        exit()
    
    pkl_files = glob.glob(os.path.join(INPUT_DIR, "*.pkl"))
    logger.info(f"🔍 Preparing to reconstruct {len(pkl_files)} files (Ultimate Mesh Mapping Version)...")
    
    for pkl_file in pkl_files:
        try:
            reconstructor.process_single_pkl(pkl_file, OUTPUT_DIR)
        except Exception as e:
            logger.error(f"❌ Error during processing: {str(e)}")
            
    logger.info("🎉 All processing complete! Your data is now perfectly standard 24-joint SMPL, go render a video and check out those perfect shoulders!")