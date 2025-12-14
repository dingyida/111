#!/usr/bin/env python3
"""
Variance Predictor Wrapper - Easy integration into your pipeline.

Usage:
    from variance_predictor_wrapper import VariancePredictorWrapper
    
    predictor = VariancePredictorWrapper('variance_predictor_final.pth')
    variance = predictor.predict(prev_pose, object_id=0, camera_id=2, resolution_id=1)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R


class VariancePredictor(nn.Module):
    """MLP with separate encoders."""
    
    def __init__(self, pose_dim=6, n_objects=5, n_cameras=5, n_resolutions=3, 
                 pose_hidden=32, embed_dim_obj=16, embed_dim_cam=16, embed_dim_res=8,
                 hidden_dim=128, dropout=0.2):
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.object_embed = nn.Embedding(n_objects, embed_dim_obj)
        self.camera_embed = nn.Embedding(n_cameras, embed_dim_cam)
        self.resolution_embed = nn.Embedding(n_resolutions, embed_dim_res)
        combined_dim = pose_hidden + embed_dim_obj + embed_dim_cam + embed_dim_res
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)
        )
    
    def forward(self, pose, obj_id, cam_id, res_id):
        pose_feat = self.pose_encoder(pose)
        obj_feat = self.object_embed(obj_id)
        cam_feat = self.camera_embed(cam_id)
        res_feat = self.resolution_embed(res_id)
        combined = torch.cat([pose_feat, obj_feat, cam_feat, res_feat], dim=1)
        return torch.exp(self.fc(combined))


class VariancePredictorWrapper:
    """
    Easy-to-use wrapper for variance prediction.
    
    Example:
        predictor = VariancePredictorWrapper('variance_predictor_final.pth')
        
        # Predict variance
        variance = predictor.predict(
            prev_pose=np.array([0, 1.2, 11, 0.3, -1.6, -0.4]),
            object_id=0,
            camera_id=2,
            resolution_id=1
        )
        
        print(f"Position uncertainty: {variance['position_std_mm']} mm")
        print(f"Rotation uncertainty: {variance['rotation_std_deg']}°")
    """
    
    def __init__(self, model_path='variance_predictor_final.pth', device=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = VariancePredictor()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Load scaler
        self.pose_mean = checkpoint['pose_scaler_mean']
        self.pose_scale = checkpoint['pose_scaler_scale']
        
        print(f"✓ Loaded variance predictor on {self.device}")
    
    def predict(self, prev_pose, object_id, camera_id, resolution_id):
        """
        Predict variance.
        
        Args:
            prev_pose: Previous pose as 4x4 matrix or 6D vector [tx,ty,tz,rx,ry,rz]
            object_id: int (0-4)
            camera_id: int (0-4)
            resolution_id: int (0=320x180, 1=640x360, 2=1280x720)
        
        Returns:
            dict with:
                - position_std: (3,) in meters
                - rotation_std: (3,) in radians
                - position_std_mm: (3,) in millimeters
                - rotation_std_deg: (3,) in degrees
        """
        # Convert pose to 6-DoF
        pose_6dof = self._to_6dof(prev_pose)
        
        # Normalize
        pose_normalized = (pose_6dof - self.pose_mean) / self.pose_scale
        
        # Convert to tensors
        pose_tensor = torch.FloatTensor(pose_normalized).unsqueeze(0).to(self.device)
        obj_tensor = torch.LongTensor([object_id]).to(self.device)
        cam_tensor = torch.LongTensor([camera_id]).to(self.device)
        res_tensor = torch.LongTensor([resolution_id]).to(self.device)
        
        # Predict
        with torch.no_grad():
            variance = self.model(pose_tensor, obj_tensor, cam_tensor, res_tensor)
        
        variance_np = variance.cpu().numpy()[0]
        
        return {
            'position_std': variance_np[:3],
            'rotation_std': variance_np[3:],
            'position_std_mm': variance_np[:3] * 1000,
            'rotation_std_deg': np.rad2deg(variance_np[3:]),
        }
    
    def predict_batch(self, prev_poses, object_ids, camera_ids, resolution_ids):
        """
        Predict variance for a batch of inputs.
        
        Args:
            prev_poses: (N, 6) or list of poses
            object_ids: (N,) or list
            camera_ids: (N,) or list
            resolution_ids: (N,) or list
        
        Returns:
            dict with batched results
        """
        # Convert all inputs to arrays
        if isinstance(prev_poses, list):
            prev_poses = np.array([self._to_6dof(p) for p in prev_poses])
        else:
            prev_poses = np.array([self._to_6dof(p) for p in prev_poses])
        
        object_ids = np.array(object_ids)
        camera_ids = np.array(camera_ids)
        resolution_ids = np.array(resolution_ids)
        
        # Normalize poses
        poses_normalized = (prev_poses - self.pose_mean) / self.pose_scale
        
        # Convert to tensors
        pose_tensor = torch.FloatTensor(poses_normalized).to(self.device)
        obj_tensor = torch.LongTensor(object_ids).to(self.device)
        cam_tensor = torch.LongTensor(camera_ids).to(self.device)
        res_tensor = torch.LongTensor(resolution_ids).to(self.device)
        
        # Predict
        with torch.no_grad():
            variance = self.model(pose_tensor, obj_tensor, cam_tensor, res_tensor)
        
        variance_np = variance.cpu().numpy()
        
        return {
            'position_std': variance_np[:, :3],
            'rotation_std': variance_np[:, 3:],
            'position_std_mm': variance_np[:, :3] * 1000,
            'rotation_std_deg': np.rad2deg(variance_np[:, 3:]),
        }
    
    def _to_6dof(self, pose):
        """Convert pose to 6-DoF vector."""
        if isinstance(pose, np.ndarray):
            if pose.shape == (4, 4):
                translation = pose[:3, 3]
                rotation = R.from_matrix(pose[:3, :3]).as_rotvec()
                return np.concatenate([translation, rotation])
            elif pose.shape == (6,):
                return pose
            else:
                raise ValueError(f"Invalid pose shape: {pose.shape}")
        else:
            raise ValueError("Pose must be numpy array")


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of the wrapper."""
    print("="*80)
    print("Variance Predictor Wrapper - Example Usage")
    print("="*80)
    print()
    
    # Initialize predictor
    predictor = VariancePredictorWrapper('variance_predictor_final.pth')
    print()
    
    # Example 1: Single prediction
    print("Example 1: Single Prediction")
    print("-" * 40)
    
    prev_pose = np.array([0.002, 1.204, 11.061, 0.325, -1.614, -0.433])
    result = predictor.predict(prev_pose, object_id=0, camera_id=0, resolution_id=2)
    
    print(f"Position std: {result['position_std_mm']} mm")
    print(f"Rotation std: {result['rotation_std_deg']}°")
    print()
    
    # Example 2: Use in a loop (tracking)
    print("Example 2: Tracking Loop Simulation")
    print("-" * 40)
    
    # Simulate 10 frames of tracking
    base_pose = np.array([0.0, 1.2, 11.0, 0.3, -1.6, -0.4])
    
    for frame_id in range(10):
        # Add some random motion
        pose = base_pose + np.random.randn(6) * 0.01
        
        # Predict variance
        variance = predictor.predict(pose, object_id=0, camera_id=2, resolution_id=1)
        
        # Use variance for decision making
        pos_uncertainty = variance['position_std_mm'].max()
        rot_uncertainty = variance['rotation_std_deg'].max()
        
        if pos_uncertainty > 5.0:  # > 5mm
            status = "⚠️  High uncertainty - need reinitialization"
        elif pos_uncertainty > 2.0:  # > 2mm
            status = "⚡ Medium uncertainty - proceed with caution"
        else:
            status = "✓ Low uncertainty - good tracking"
        
        print(f"Frame {frame_id:2d}: pos={pos_uncertainty:.2f}mm, "
              f"rot={rot_uncertainty:.2f}° | {status}")
    
    print()
    
    # Example 3: Batch prediction
    print("Example 3: Batch Prediction (5 poses)")
    print("-" * 40)
    
    poses = [base_pose + np.random.randn(6) * 0.01 for _ in range(5)]
    object_ids = [0, 0, 1, 1, 0]
    camera_ids = [0, 1, 2, 3, 4]
    resolution_ids = [2, 2, 1, 1, 0]
    
    results = predictor.predict_batch(poses, object_ids, camera_ids, resolution_ids)
    
    print(f"Position std (mm):")
    print(results['position_std_mm'])
    print(f"\nRotation std (°):")
    print(results['rotation_std_deg'])
    print()
    
    print("="*80)


if __name__ == '__main__':
    main()
