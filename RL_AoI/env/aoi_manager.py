#!/usr/bin/env python3
"""
AoI Manager for tracking Age of Information state.

Manages:
- Pose AoI: Increases every frame, decreases on observation
- Style AoI: Decreases on observation (computed from observation history)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import math


def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n uniform points on unit sphere using Fibonacci lattice."""
    golden_angle = math.pi * (3 - math.sqrt(5))
    indices = np.arange(n)
    z = (2 * indices + 1) / n - 1
    radius = np.sqrt(1 - z * z)
    theta = golden_angle * indices
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    return pts


@dataclass
class ObservationRecord:
    """Record of a single observation for style error computation."""
    frame: int
    camera_id: int
    resolution_id: int
    camera_position: np.ndarray
    object_position: np.ndarray
    width: int
    height: int
    fov_deg: float = 45.0


class AoIManager:
    """
    Manages Age of Information (AoI) state for all objects.
    
    Pose AoI:
        - Starts at 0
        - Increases by fixed amount each frame (object-dependent)
        - On observation: pose_aoi = min(current, variance_6dof)
    
    Style AoI:
        - Starts at 1
        - On observation: recomputed from cumulative observation history
        - Monotonically decreasing
        - Formula: Q = 1 - exp(-k * mean_coverage), style_error = 1 - Q
    """
    
    def __init__(
        self,
        num_objects: int = 5,
        pose_aoi_increments: Optional[List[float]] = None,
        initial_pose_aoi: float = 0.0,
        initial_style_aoi: float = 1.0,
        style_error_params: Optional[Dict] = None
    ):
        """
        Initialize the AoI manager.
        
        Args:
            num_objects: Number of objects to track
            pose_aoi_increments: List of per-frame increments for each object
            initial_pose_aoi: Initial pose AoI value
            initial_style_aoi: Initial style AoI value
            style_error_params: Parameters for style error computation
        """
        self.num_objects = num_objects
        
        # Default pose AoI increments
        if pose_aoi_increments is None:
            self.pose_aoi_increments = np.array([0.006, 0.02, 0.05, 0.01, 0.036])
        else:
            self.pose_aoi_increments = np.array(pose_aoi_increments)
        
        self.initial_pose_aoi = initial_pose_aoi
        self.initial_style_aoi = initial_style_aoi
        
        # Style error parameters
        if style_error_params is None:
            self.style_error_params = {
                'k': 4.0,
                'p': 8.0,
                'alpha': 0.2,
                'beta': 0.6,
                'zeta': 1.0,
                'default_fov_deg': 45.0,
                'n_samples': 1024
            }
        else:
            self.style_error_params = style_error_params
        
        # Pre-compute sphere samples for efficiency
        n_samples = self.style_error_params.get('n_samples', 1024)
        self.sphere_samples = fibonacci_sphere(n_samples)
        
        # State arrays
        self.pose_aoi = np.zeros(num_objects)
        self.style_aoi = np.ones(num_objects)
        
        # Observation history per object
        self.observation_history: List[List[ObservationRecord]] = [[] for _ in range(num_objects)]
        
        self.reset()
    
    def reset(self):
        """Reset all AoI values and observation history."""
        self.pose_aoi = np.full(self.num_objects, self.initial_pose_aoi)
        self.style_aoi = np.full(self.num_objects, self.initial_style_aoi)
        self.observation_history = [[] for _ in range(self.num_objects)]
    
    def increment_pose_aoi(self):
        """Increment pose AoI for all objects by their fixed amounts."""
        self.pose_aoi += self.pose_aoi_increments
    
    def update_pose_aoi(self, object_id: int, variance_6dof: float):
        """
        Update pose AoI for a single object after observation.
        
        Args:
            object_id: Object ID (0-4)
            variance_6dof: 6-DoF variance from variance predictor
        """
        self.pose_aoi[object_id] = min(self.pose_aoi[object_id], variance_6dof)
    
    def add_observation(
        self,
        object_id: int,
        frame: int,
        camera_id: int,
        resolution_id: int,
        camera_position: np.ndarray,
        object_position: np.ndarray,
        width: int,
        height: int,
        fov_deg: float = 45.0
    ):
        """
        Add an observation record for style error computation.
        
        Args:
            object_id: Object ID
            frame: Frame number
            camera_id: Camera ID
            resolution_id: Resolution ID (1 or 2)
            camera_position: 3D camera position
            object_position: 3D object position
            width: Image width
            height: Image height
            fov_deg: Camera field of view in degrees
        """
        record = ObservationRecord(
            frame=frame,
            camera_id=camera_id,
            resolution_id=resolution_id,
            camera_position=camera_position.copy(),
            object_position=object_position.copy(),
            width=width,
            height=height,
            fov_deg=fov_deg
        )
        self.observation_history[object_id].append(record)
    
    def compute_style_error(self, object_id: int) -> float:
        """
        Compute style error for an object based on observation history.
        
        Formula:
            Q = 1 - exp(-k * mean_coverage)
            style_error = 1 - Q = exp(-k * mean_coverage)
        
        Args:
            object_id: Object ID
        
        Returns:
            Style error value in [0, 1]
        """
        history = self.observation_history[object_id]
        
        if len(history) == 0:
            return self.initial_style_aoi
        
        # Build arrays from observation history
        camera_positions = np.array([obs.camera_position for obs in history])
        object_positions = np.array([obs.object_position for obs in history])
        widths = np.array([obs.width for obs in history])
        heights = np.array([obs.height for obs in history])
        fov_degs = np.array([obs.fov_deg for obs in history])
        
        # Get parameters
        params = self.style_error_params
        k = params.get('k', 4.0)
        p = params.get('p', 8.0)
        alpha = params.get('alpha', 0.2)
        beta = params.get('beta', 0.6)
        zeta = params.get('zeta', 1.0)
        
        # Compute viewing directions and distances
        vec = object_positions - camera_positions
        d_i = np.linalg.norm(vec, axis=1)
        omega_i = vec / (d_i[:, None] + 1e-12)
        
        # Compute resolution terms (pixels per steradian)
        pixels = widths * heights
        fov_rad = np.radians(fov_degs)
        solid_angles = 2 * np.pi * (1 - np.cos(fov_rad))
        solid_angles = np.clip(solid_angles, 1e-8, None)
        
        pps = pixels / solid_angles
        median_pps = np.median(pps) if np.any(pps > 0) else 1.0
        r_i = pps / (median_pps if median_pps > 0 else 1.0)
        
        # Per-camera weight: w_i = (r_i^beta) / (1 + alpha * d_i^2)^zeta
        w_i = (r_i ** beta) / ((1.0 + alpha * (d_i ** 2)) ** zeta)
        w_i = np.clip(w_i, 0.0, 1.0)
        
        # Compute coverage at each sphere sample point
        phi_i = fov_rad
        phi_safe = np.clip(phi_i, 1e-6, None)[None, :]
        
        # Angular distance from each sample to each observation direction
        dot = np.clip(self.sphere_samples @ omega_i.T, -1.0, 1.0)
        gamma = np.arccos(dot)  # (n_samples, n_obs)
        
        # Contribution: a_i = w_i * exp(-(gamma / phi)^p)
        a = w_i[None, :] * np.exp(-(gamma / phi_safe) ** p)
        a = np.clip(a, 0.0, 1.0)
        
        # Soft union over all observations: C = 1 - prod(1 - a)
        C = 1.0 - np.prod(1.0 - a, axis=1)
        
        # Mean coverage over sphere
        mean_coverage = float(np.mean(C))
        
        # Final style error
        Q = 1.0 - math.exp(-k * mean_coverage)
        style_error = 1.0 - Q
        
        return style_error
    
    def update_style_aoi(self, object_id: int):
        """
        Update style AoI for an object based on current observation history.
        
        Style AoI can only decrease (monotonically decreasing).
        
        Args:
            object_id: Object ID
        """
        new_style_error = self.compute_style_error(object_id)
        self.style_aoi[object_id] = min(self.style_aoi[object_id], new_style_error)
    
    def update_all_style_aoi(self):
        """Update style AoI for all objects."""
        for obj_id in range(self.num_objects):
            self.update_style_aoi(obj_id)
    
    def get_total_aoi(self, alpha: float = 0.5) -> float:
        """
        Get the weighted total AoI.
        
        Args:
            alpha: Weight for pose AoI (1-alpha for style AoI)
        
        Returns:
            Total weighted AoI
        """
        total_pose = np.sum(self.pose_aoi)
        total_style = np.sum(self.style_aoi)
        return alpha * total_pose + (1 - alpha) * total_style
    
    def get_pose_aoi(self) -> np.ndarray:
        """Get current pose AoI for all objects."""
        return self.pose_aoi.copy()
    
    def get_style_aoi(self) -> np.ndarray:
        """Get current style AoI for all objects."""
        return self.style_aoi.copy()
    
    def get_observation_counts(self) -> np.ndarray:
        """Get number of observations per object."""
        return np.array([len(hist) for hist in self.observation_history])
    
    def get_camera_observation_counts(self) -> np.ndarray:
        """Get number of observations per camera (across all objects)."""
        counts = np.zeros(5)
        for obj_history in self.observation_history:
            for obs in obj_history:
                if obs.camera_id < len(counts):
                    counts[obs.camera_id] += 1
        return counts


def test_aoi_manager():
    """Test the AoI manager."""
    manager = AoIManager()
    
    print("Initial state:")
    print(f"  Pose AoI: {manager.get_pose_aoi()}")
    print(f"  Style AoI: {manager.get_style_aoi()}")
    
    # Simulate 10 frames
    print("\nAfter 10 frames without observation:")
    for _ in range(10):
        manager.increment_pose_aoi()
    print(f"  Pose AoI: {manager.get_pose_aoi()}")
    
    # Add observation
    print("\nAdding observation for object 0 from camera 0:")
    manager.add_observation(
        object_id=0, frame=10, camera_id=0, resolution_id=1,
        camera_position=np.array([5.0, 0.0, 2.0]),
        object_position=np.array([0.0, 1.0, 11.0]),
        width=640, height=360, fov_deg=45.0
    )
    manager.update_pose_aoi(0, variance_6dof=0.01)
    manager.update_style_aoi(0)
    
    print(f"  Pose AoI: {manager.get_pose_aoi()}")
    print(f"  Style AoI: {manager.get_style_aoi()}")
    
    # Add more observations from different cameras
    print("\nAdding observations from 4 more cameras:")
    for cam_id in range(1, 5):
        angle = 2 * np.pi * cam_id / 5
        cam_pos = np.array([5.0 * np.cos(angle), 5.0 * np.sin(angle), 2.0])
        manager.add_observation(
            object_id=0, frame=10+cam_id, camera_id=cam_id, resolution_id=1,
            camera_position=cam_pos,
            object_position=np.array([0.0, 1.0, 11.0]),
            width=640, height=360, fov_deg=45.0
        )
    
    manager.update_style_aoi(0)
    print(f"  Style AoI after 5 diverse observations: {manager.get_style_aoi()}")


if __name__ == "__main__":
    test_aoi_manager()
