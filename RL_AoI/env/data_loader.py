#!/usr/bin/env python3
"""
Data Loader for AoI RL Environment.

Loads pose sequences from render_output folder structure:
    render_output/
    ├── fp_cam{0-4}_{resolution}/
    │   ├── annotated_poses/
    │   │   ├── {frame}_{actor}.txt  →  4×4 pose matrix
    │   ├── cam_K.txt                →  camera intrinsics
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R


class PoseDataLoader:
    """
    Loads and manages pose data from render_output directory.
    
    Attributes:
        num_cameras: Number of cameras (default 5)
        num_objects: Number of objects (default 5)
        num_frames: Number of frames in the sequence
        object_mapping: Dict mapping actor names to object IDs
    """
    
    def __init__(
        self,
        render_output_path: str,
        num_cameras: int = 5,
        num_objects: int = 5,
        object_mapping: Optional[Dict[str, int]] = None,
        default_resolution: str = "640x360"
    ):
        """
        Initialize the data loader.
        
        Args:
            render_output_path: Path to render_output directory
            num_cameras: Number of cameras
            num_objects: Number of objects
            object_mapping: Dict mapping actor names (e.g., "actor197") to object IDs (0-4)
            default_resolution: Default resolution folder to use for loading poses
        """
        self.render_output_path = render_output_path
        self.num_cameras = num_cameras
        self.num_objects = num_objects
        self.default_resolution = default_resolution
        
        # Default object mapping
        if object_mapping is None:
            self.object_mapping = {
                "actor197": 0,
                "actor198": 1,
                "actor199": 2,
                "actor200": 3,
                "actor201": 4
            }
        else:
            self.object_mapping = object_mapping
        
        # Reverse mapping: object_id -> actor_name
        self.id_to_actor = {v: k for k, v in self.object_mapping.items()}
        
        # Storage for loaded data
        self.poses: Dict[Tuple[int, int, int], np.ndarray] = {}  # (frame, camera, object) -> 4x4 matrix
        self.camera_intrinsics: Dict[int, np.ndarray] = {}  # camera_id -> 3x3 matrix
        self.camera_positions: Dict[int, np.ndarray] = {}  # camera_id -> [x, y, z]
        
        # Load data
        self._scan_and_load()
    
    def _scan_and_load(self):
        """Scan directory structure and load all pose data."""
        if not os.path.exists(self.render_output_path):
            print(f"Warning: render_output path not found: {self.render_output_path}")
            print("Using synthetic data for testing.")
            self._generate_synthetic_data()
            return
        
        # Find all camera folders
        camera_folders = []
        for item in os.listdir(self.render_output_path):
            if item.startswith("fp_cam") and os.path.isdir(os.path.join(self.render_output_path, item)):
                camera_folders.append(item)
        
        if not camera_folders:
            print("Warning: No camera folders found. Using synthetic data.")
            self._generate_synthetic_data()
            return
        
        # Parse camera folders and load data
        frames_found = set()
        
        for folder in camera_folders:
            # Parse folder name: fp_cam{id}_{resolution}
            match = re.match(r"fp_cam(\d+)_(\d+x\d+)", folder)
            if not match:
                continue
            
            camera_id = int(match.group(1))
            resolution = match.group(2)
            
            folder_path = os.path.join(self.render_output_path, folder)
            
            # Load camera intrinsics
            cam_k_path = os.path.join(folder_path, "cam_K.txt")
            if os.path.exists(cam_k_path) and camera_id not in self.camera_intrinsics:
                self.camera_intrinsics[camera_id] = self._load_camera_intrinsics(cam_k_path)
            
            # Load poses from annotated_poses
            poses_path = os.path.join(folder_path, "annotated_poses")
            if not os.path.exists(poses_path):
                continue
            
            for pose_file in os.listdir(poses_path):
                if not pose_file.endswith(".txt"):
                    continue
                
                # Parse filename: {frame}_{actor}.txt
                match = re.match(r"(\d+)_(actor\d+)\.txt", pose_file)
                if not match:
                    continue
                
                frame = int(match.group(1))
                actor = match.group(2)
                
                if actor not in self.object_mapping:
                    continue
                
                object_id = self.object_mapping[actor]
                
                # Load pose matrix
                pose_path = os.path.join(poses_path, pose_file)
                pose_matrix = self._load_matrix(pose_path)
                
                # Store pose (use first resolution found for each camera)
                key = (frame, camera_id, object_id)
                if key not in self.poses:
                    self.poses[key] = pose_matrix
                
                frames_found.add(frame)
        
        # Determine number of frames
        if frames_found:
            self.num_frames = max(frames_found)
            self.min_frame = min(frames_found)
        else:
            self.num_frames = 200
            self.min_frame = 1
        
        # Extract camera positions from poses (assume object position is roughly constant)
        self._estimate_camera_positions()
        
        print(f"Loaded {len(self.poses)} poses from {len(camera_folders)} camera folders")
        print(f"Frame range: {self.min_frame} to {self.num_frames}")
    
    def _load_matrix(self, path: str) -> np.ndarray:
        """Load a matrix from a text file."""
        data = np.loadtxt(path)
        # Handle different formats
        if data.ndim == 1:
            # Flat array - try to reshape to 3x3 or 4x4
            if data.size == 9:
                return data.reshape(3, 3)
            elif data.size == 12:
                # 3x4 matrix (rotation + translation)
                return data.reshape(3, 4)
            elif data.size == 16:
                return data.reshape(4, 4)
            else:
                # Return as-is if unknown size
                return data
        return data
    
    def _load_camera_intrinsics(self, path: str) -> np.ndarray:
        """Load camera intrinsics matrix (3x3)."""
        data = np.loadtxt(path)
        if data.ndim == 1:
            if data.size == 9:
                return data.reshape(3, 3)
            elif data.size >= 4:
                # Assume [fx, fy, cx, cy] format
                fx, fy, cx, cy = data[0], data[1], data[2], data[3]
                return np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            else:
                # Return identity if unknown
                return np.eye(3)
        elif data.ndim == 2:
            return data[:3, :3] if data.shape[0] >= 3 else data
        return np.eye(3)
    
    def _estimate_camera_positions(self):
        """Estimate camera positions from loaded poses."""
        # For now, use placeholder positions
        # In reality, these should come from the camera calibration
        for cam_id in range(self.num_cameras):
            # Placeholder: cameras arranged in a circle
            angle = 2 * np.pi * cam_id / self.num_cameras
            radius = 5.0
            self.camera_positions[cam_id] = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                2.0  # Height
            ])
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing when real data is not available."""
        self.num_frames = 200
        self.min_frame = 1
        
        # Generate random poses for each frame, camera, object combination
        np.random.seed(42)
        
        for frame in range(1, self.num_frames + 1):
            for cam_id in range(self.num_cameras):
                for obj_id in range(self.num_objects):
                    # Create a random pose matrix
                    translation = np.array([
                        np.random.uniform(-1, 1),
                        np.random.uniform(0.5, 1.5),
                        np.random.uniform(10, 12)
                    ])
                    # Add small motion over time
                    translation += 0.01 * frame * np.array([0.01, 0.005, -0.002])
                    
                    rotation = R.from_euler('xyz', np.random.uniform(-0.5, 0.5, 3)).as_matrix()
                    
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = translation
                    
                    self.poses[(frame, cam_id, obj_id)] = pose
        
        # Generate camera positions
        for cam_id in range(self.num_cameras):
            angle = 2 * np.pi * cam_id / self.num_cameras
            radius = 5.0
            self.camera_positions[cam_id] = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                2.0
            ])
        
        # Generate camera intrinsics (placeholder)
        for cam_id in range(self.num_cameras):
            fx, fy = 500, 500
            cx, cy = 320, 180
            self.camera_intrinsics[cam_id] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        
        print(f"Generated synthetic data: {self.num_frames} frames, {self.num_cameras} cameras, {self.num_objects} objects")
    
    def get_pose(self, frame: int, camera_id: int, object_id: int) -> np.ndarray:
        """
        Get the 4x4 pose matrix for a specific frame, camera, and object.
        
        Args:
            frame: Frame number (1-indexed)
            camera_id: Camera ID (0-4)
            object_id: Object ID (0-4)
        
        Returns:
            4x4 pose matrix
        """
        key = (frame, camera_id, object_id)
        if key in self.poses:
            return self.poses[key].copy()
        
        # Fallback: try to find pose from any camera for this frame/object
        for cam in range(self.num_cameras):
            fallback_key = (frame, cam, object_id)
            if fallback_key in self.poses:
                return self.poses[fallback_key].copy()
        
        # Return identity if not found
        return np.eye(4)
    
    def get_all_object_poses(self, frame: int, camera_id: int = 0) -> Dict[int, np.ndarray]:
        """
        Get poses for all objects at a specific frame.
        
        Args:
            frame: Frame number
            camera_id: Camera ID to use for pose lookup
        
        Returns:
            Dict mapping object_id to 4x4 pose matrix
        """
        poses = {}
        for obj_id in range(self.num_objects):
            poses[obj_id] = self.get_pose(frame, camera_id, obj_id)
        return poses
    
    def get_object_position(self, frame: int, object_id: int, camera_id: int = 0) -> np.ndarray:
        """
        Get the 3D position of an object at a specific frame.
        
        Args:
            frame: Frame number
            object_id: Object ID
            camera_id: Camera ID for pose lookup
        
        Returns:
            3D position array [x, y, z]
        """
        pose = self.get_pose(frame, camera_id, object_id)
        return pose[:3, 3].copy()
    
    def get_camera_position(self, camera_id: int) -> np.ndarray:
        """
        Get the 3D position of a camera.
        
        Args:
            camera_id: Camera ID
        
        Returns:
            3D position array [x, y, z]
        """
        if camera_id in self.camera_positions:
            return self.camera_positions[camera_id].copy()
        return np.zeros(3)
    
    def pose_to_6dof(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert 4x4 pose matrix to 6-DoF vector [tx, ty, tz, rx, ry, rz].
        
        Args:
            pose: 4x4 pose matrix
        
        Returns:
            6D vector [translation (3), rotation_vector (3)]
        """
        translation = pose[:3, 3]
        rotation = R.from_matrix(pose[:3, :3]).as_rotvec()
        return np.concatenate([translation, rotation])
    
    def get_pose_6dof(self, frame: int, camera_id: int, object_id: int) -> np.ndarray:
        """
        Get the 6-DoF pose vector for a specific frame, camera, and object.
        
        Returns:
            6D vector [tx, ty, tz, rx, ry, rz]
        """
        pose = self.get_pose(frame, camera_id, object_id)
        return self.pose_to_6dof(pose)


def test_data_loader():
    """Test the data loader with synthetic data."""
    loader = PoseDataLoader(
        render_output_path="./nonexistent",  # Will use synthetic data
        num_cameras=5,
        num_objects=5
    )
    
    print(f"\nTest: Get pose for frame 1, camera 0, object 0")
    pose = loader.get_pose(1, 0, 0)
    print(f"Pose shape: {pose.shape}")
    print(f"Pose:\n{pose}")
    
    print(f"\nTest: Get 6-DoF pose")
    pose_6dof = loader.get_pose_6dof(1, 0, 0)
    print(f"6-DoF: {pose_6dof}")
    
    print(f"\nTest: Get all object poses for frame 50")
    all_poses = loader.get_all_object_poses(50)
    print(f"Number of objects: {len(all_poses)}")
    
    print(f"\nTest: Get camera position")
    cam_pos = loader.get_camera_position(0)
    print(f"Camera 0 position: {cam_pos}")


if __name__ == "__main__":
    test_data_loader()
