#!/usr/bin/env python3
"""
Generate variance training data by running FoundationPose on all combinations.

This script:
1. Loops through 5 objects ¡Á 5 cameras ¡Á 3 resolutions = 75 combinations
2. Runs FoundationPose (with distribution output)
3. Collects: previous pose, object ID, camera ID, resolution ID, variance
4. Saves to CSV for training
"""

import os
import sys
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation as R
import glob
import logging
import cv2
import imageio

# Add FoundationPose to path
sys.path.append('/home/yid324/FoundationPose')

from estimater_distribution import FoundationPose, ScorePredictor, PoseRefinePredictor
from datareader import *
import nvdiffrast.torch as dr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
ROOT_DIR = '/home/yid324/FoundationPose/render_output'
MESH_K = '/home/yid324/FoundationPose/mesh_k/textured_mesh.obj'
MESH_M = '/home/yid324/FoundationPose/mesh_m/textured_simple.obj'

# Object assignments
OBJECT_MESHES = {
    197: (MESH_K, 0),  # Object 1 (ID=0 for training)
    198: (MESH_K, 1),  # Object 2 (ID=1)
    199: (MESH_K, 2),  # Object 3 (ID=2)
    200: (MESH_M, 3),  # Object 4 (ID=3)
    201: (MESH_M, 4),  # Object 5 (ID=4)
}

# Camera and resolution mappings
CAMERA_IDS = {
    'fp_cam0': 0,
    'fp_cam1': 1,
    'fp_cam2': 2,
    'fp_cam3': 3,
    'fp_cam4': 4,
}

RESOLUTION_IDS = {
    '320x180': 0,
    '640x360': 1,
    '1280x720': 2,
}


def parse_folder_name(folder_name):
    """
    Parse folder name like 'fp_cam0_320x180' into camera and resolution IDs.
    
    Returns:
        camera_id (int), resolution_id (int)
    """
    parts = folder_name.split('_')
    camera_name = f"{parts[0]}_{parts[1]}"  # fp_cam0
    resolution_name = parts[2]  # 320x180
    
    camera_id = CAMERA_IDS[camera_name]
    resolution_id = RESOLUTION_IDS[resolution_name]
    
    return camera_id, resolution_id


def load_pose_file(filepath):
    """Load 4x4 pose matrix from text file."""
    pose = np.loadtxt(filepath)
    if pose.shape == (4, 4):
        return pose
    elif pose.shape == (16,):
        return pose.reshape(4, 4)
    else:
        raise ValueError(f"Unexpected pose shape: {pose.shape}")


def pose_to_6dof(pose):
    """
    Convert 4x4 pose matrix to 6-DoF vector.
    
    Returns:
        (6,) array: [tx, ty, tz, rx, ry, rz]
    """
    translation = pose[:3, 3]
    rotation = R.from_matrix(pose[:3, :3]).as_rotvec()
    return np.concatenate([translation, rotation])


def extract_object_id_from_filename(filename):
    """
    Extract object ID from filename like '000001_actor197.txt'.
    
    Returns:
        actor_id (int): e.g., 197, 198, etc.
    """
    # Remove extension and split
    name = os.path.splitext(filename)[0]
    parts = name.split('_actor')
    if len(parts) == 2:
        return int(parts[1])
    else:
        raise ValueError(f"Cannot parse object ID from: {filename}")


class CustomReader:
    """Custom data reader for your rendered sequences."""
    
    def __init__(self, folder_path, actor_id):
        """
        Args:
            folder_path: Path to sequence folder (e.g., render_output/fp_cam0_320x180)
            actor_id: Actor ID (197-201) to load masks for
        """
        self.folder_path = folder_path
        self.actor_id = actor_id
        
        # Load camera intrinsics
        cam_k_file = os.path.join(folder_path, 'cam_K.txt')
        self.K = np.loadtxt(cam_k_file).reshape(3, 3)
        
        # Get file lists (RGB and depth are in main folders)
        self.rgb_files = sorted(glob.glob(os.path.join(folder_path, 'rgb', '*.png')))
        self.depth_files = sorted(glob.glob(os.path.join(folder_path, 'depth', '*.png')))
        
        # Masks are in actor-specific subfolder
        mask_actor_dir = os.path.join(folder_path, 'masks', f'actor{actor_id}')
        if os.path.exists(mask_actor_dir):
            self.mask_files = sorted(glob.glob(os.path.join(mask_actor_dir, '*.png')))
            logging.info(f"Found {len(self.mask_files)} mask files in {mask_actor_dir}")
            if len(self.mask_files) > 0:
                logging.info(f"  First mask: {os.path.basename(self.mask_files[0])}")
                logging.info(f"  Last mask: {os.path.basename(self.mask_files[-1])}")
        else:
            logging.warning(f"Mask folder not found: {mask_actor_dir}")
            self.mask_files = []
        
        # Check for mismatches and use minimum length
        if len(self.rgb_files) != len(self.depth_files):
            logging.warning(f"File count mismatch in {folder_path}:")
            logging.warning(f"  RGB: {len(self.rgb_files)}, Depth: {len(self.depth_files)}")
            logging.warning(f"  Using minimum length: {min(len(self.rgb_files), len(self.depth_files))}")
            min_len = min(len(self.rgb_files), len(self.depth_files))
            self.rgb_files = self.rgb_files[:min_len]
            self.depth_files = self.depth_files[:min_len]
        
        # Get image dimensions from first RGB
        if len(self.rgb_files) > 0:
            first_rgb = cv2.imread(self.rgb_files[0])
            self.H, self.W = first_rgb.shape[:2]
            logging.info(f"  First RGB: {os.path.basename(self.rgb_files[0])}")
        else:
            raise ValueError(f"No RGB files found in {folder_path}")
        
        logging.info(f"Loaded {len(self.rgb_files)} frames from {folder_path} for actor{actor_id}")
        logging.info(f"  RGB files: {len(self.rgb_files)}")
        logging.info(f"  Depth files: {len(self.depth_files)}")
        logging.info(f"  Mask files: {len(self.mask_files)}")
    
    def __len__(self):
        return len(self.rgb_files)
    
    def get_color(self, i):
        """Get RGB image at index i."""
        return imageio.imread(self.rgb_files[i])[..., :3]
    
    def get_depth(self, i):
        """Get depth image at index i."""
        depth = cv2.imread(self.depth_files[i], -1) / 1000.0  # Convert to meters
        return depth
    
    def get_mask(self, i):
        """
        Get mask at index i.
        
        The mask files might have different numbering than RGB/depth.
        We'll use index-based access assuming they're in the same order.
        """
        if i < len(self.mask_files):
            mask = cv2.imread(self.mask_files[i], -1)
            if mask is None:
                raise ValueError(f"Failed to load mask: {self.mask_files[i]}")
            if len(mask.shape) == 3:
                mask = mask[..., 0]
            return (mask > 0).astype(np.uint8)
        else:
            raise IndexError(f"Mask index {i} out of range (only {len(self.mask_files)} masks available)")


def run_foundation_pose_on_sequence(folder_path, mesh_file, actor_id, object_id, camera_id, resolution_id):
    """
    Run FoundationPose on a sequence and collect variance data.
    
    Args:
        folder_path: Path to data folder (e.g., 'render_output/fp_cam0_320x180')
        mesh_file: Path to mesh file
        actor_id: Actor ID (197-201) for loading masks
        object_id: Object ID (0-4) for training
        camera_id: Camera ID (0-4)
        resolution_id: Resolution ID (0-2)
    
    Returns:
        List of training samples (dicts)
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing: Actor {actor_id} ¡ú Object {object_id}, Camera {camera_id}, Resolution {resolution_id}")
    logging.info(f"Folder: {folder_path}")
    logging.info(f"Mesh: {mesh_file}")
    logging.info(f"{'='*80}\n")
    
    # Load mesh
    mesh = trimesh.load(mesh_file)
    
    # Initialize FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir='/tmp/fp_debug',
        debug=0,  # Minimal debug output
        glctx=glctx,
        distribution_top_k=20
    )
    
    # Load data with actor_id for masks
    reader = CustomReader(folder_path, actor_id)
    
    # Storage for training samples
    training_samples = []
    
    # Previous pose (starts as None)
    prev_pose = None
    
    # Process frames
    for i in range(len(reader)):
        try:
            color = reader.get_color(i)
            depth = reader.get_depth(i)
            
            if i == 0:
                # Register first frame
                mask = reader.get_mask(0)
                pose, pose_dist = est.register(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=5,
                    return_distribution=True
                )
                prev_pose = pose
                logging.info(f"  Frame {i}: Registered")
            else:
                # Track subsequent frames
                pose, pose_dist = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=reader.K,
                    iteration=2,
                    return_distribution=True
                )
                
                # Extract variance
                pos_std = pose_dist.get_position_std()
                rot_std = pose_dist.get_rotation_std()
                
                # Convert previous pose to 6-DoF
                prev_6dof = pose_to_6dof(prev_pose)
                
                # Create training sample
                sample = {
                    # Previous pose (6D)
                    'prev_tx': prev_6dof[0],
                    'prev_ty': prev_6dof[1],
                    'prev_tz': prev_6dof[2],
                    'prev_rx': prev_6dof[3],
                    'prev_ry': prev_6dof[4],
                    'prev_rz': prev_6dof[5],
                    # Context
                    'object_id': object_id,
                    'camera_id': camera_id,
                    'resolution_id': resolution_id,
                    # Labels (variance)
                    'var_pos_x': pos_std[0],
                    'var_pos_y': pos_std[1],
                    'var_pos_z': pos_std[2],
                    'var_rot_x': rot_std[0],
                    'var_rot_y': rot_std[1],
                    'var_rot_z': rot_std[2],
                }
                
                training_samples.append(sample)
                
                # Update prev_pose for next iteration
                prev_pose = pose
                
                if i % 20 == 0:
                    logging.info(f"  Frame {i}/{len(reader)}: "
                               f"pos_std={pos_std.mean()*1000:.3f}mm, "
                               f"rot_std={np.rad2deg(rot_std.mean()):.3f}¡ã")
            
        except Exception as e:
            logging.error(f"  Error processing frame {i}: {e}")
            continue
    
    logging.info(f"  Completed: Collected {len(training_samples)} samples\n")
    return training_samples


def main():
    """Main function to generate all training data."""
    
    all_training_data = []
    
    # Get all folders
    folders = sorted([f for f in os.listdir(ROOT_DIR) 
                     if os.path.isdir(os.path.join(ROOT_DIR, f)) and f.startswith('fp_cam')])
    
    logging.info(f"Found {len(folders)} folders to process")
    
    # Process each object
    for actor_id, (mesh_file, training_obj_id) in OBJECT_MESHES.items():
        logging.info(f"\n{'#'*80}")
        logging.info(f"# OBJECT {training_obj_id} (Actor {actor_id})")
        logging.info(f"# Mesh: {mesh_file}")
        logging.info(f"{'#'*80}\n")
        
        # Process each folder (camera ¡Á resolution combination)
        for folder_name in folders:
            folder_path = os.path.join(ROOT_DIR, folder_name)
            
            # Parse camera and resolution IDs
            camera_id, resolution_id = parse_folder_name(folder_name)
            
            # Check if this folder has data for this object
            # (Check if annotated_poses has files with this actor_id)
            pose_dir = os.path.join(folder_path, 'annotated_poses')
            pose_files = glob.glob(os.path.join(pose_dir, f'*_actor{actor_id}.txt'))
            
            if len(pose_files) == 0:
                logging.info(f"Skipping {folder_name} - no data for actor {actor_id}")
                continue
            
            # Run FoundationPose and collect data
            try:
                samples = run_foundation_pose_on_sequence(
                    folder_path=folder_path,
                    mesh_file=mesh_file,
                    actor_id=actor_id,
                    object_id=training_obj_id,
                    camera_id=camera_id,
                    resolution_id=resolution_id
                )
                all_training_data.extend(samples)
                
            except Exception as e:
                logging.error(f"Failed to process {folder_name} for object {training_obj_id}: {e}")
                continue
    
    # Save to CSV
    if len(all_training_data) > 0:
        df = pd.DataFrame(all_training_data)
        output_file = 'variance_training_data.csv'
        df.to_csv(output_file, index=False)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"SUCCESS! Saved {len(df)} training samples to {output_file}")
        logging.info(f"{'='*80}\n")
        
        # Print statistics
        logging.info("\nDataset Statistics:")
        logging.info(f"  Total samples: {len(df)}")
        logging.info(f"  Objects: {df['object_id'].unique()}")
        logging.info(f"  Cameras: {df['camera_id'].unique()}")
        logging.info(f"  Resolutions: {df['resolution_id'].unique()}")
        logging.info(f"\nSamples per object:")
        logging.info(df['object_id'].value_counts().sort_index())
        logging.info(f"\nSamples per camera:")
        logging.info(df['camera_id'].value_counts().sort_index())
        logging.info(f"\nSamples per resolution:")
        logging.info(df['resolution_id'].value_counts().sort_index())
        
        # Variance statistics
        logging.info(f"\nVariance Statistics:")
        logging.info(f"  Position std (mm): mean={df[['var_pos_x','var_pos_y','var_pos_z']].mean().mean()*1000:.3f}, "
                    f"std={df[['var_pos_x','var_pos_y','var_pos_z']].std().mean()*1000:.3f}")
        logging.info(f"  Rotation std (deg): mean={np.rad2deg(df[['var_rot_x','var_rot_y','var_rot_z']].mean().mean()):.3f}, "
                    f"std={np.rad2deg(df[['var_rot_x','var_rot_y','var_rot_z']].std().mean()):.3f}")
        
    else:
        logging.error("No training data collected!")


if __name__ == '__main__':
    main()
