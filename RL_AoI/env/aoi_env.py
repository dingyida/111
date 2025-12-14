#!/usr/bin/env python3
"""
AoI Gym Environment for Reinforcement Learning.

Environment for minimizing Age of Information (AoI) through camera selection.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

from env.data_loader import PoseDataLoader
from env.aoi_manager import AoIManager


def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n uniform points on unit sphere using Fibonacci lattice."""
    import math
    golden_angle = math.pi * (3 - math.sqrt(5))
    indices = np.arange(n)
    z = (2 * indices + 1) / n - 1
    radius = np.sqrt(1 - z * z)
    theta = golden_angle * indices
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y, z], axis=1)


class SphericalCoverageEncoder:
    """
    Encodes spherical coverage from observation history into a fixed-size vector.
    
    The encoder:
    1. Samples points on a unit sphere
    2. Computes coverage at each point based on observation history
    3. Applies learned/fixed basis functions to produce embedding
    
    This allows the RL agent to understand which parts of the viewing sphere
    have been covered and which need more observations.
    """
    
    def __init__(
        self,
        n_samples: int = 64,
        embed_dim: int = 16,
        fov_deg: float = 45.0,
        decay_power: float = 8.0
    ):
        """
        Initialize the spherical coverage encoder.
        
        Args:
            n_samples: Number of sample points on sphere
            embed_dim: Output embedding dimension
            fov_deg: Camera field of view in degrees
            decay_power: Angular decay exponent
        """
        self.n_samples = n_samples
        self.embed_dim = embed_dim
        self.fov_rad = np.radians(fov_deg)
        self.decay_power = decay_power
        
        # Generate fixed sample points on sphere
        self.sphere_points = fibonacci_sphere(n_samples)  # (n_samples, 3)
        
        # Create projection matrix: (n_samples,) -> (embed_dim,)
        # Use random projection (could be learned in more advanced version)
        np.random.seed(42)  # Fixed seed for reproducibility
        self.projection = self._create_projection_matrix()
    
    def _create_projection_matrix(self) -> np.ndarray:
        """
        Create projection matrix to convert coverage vector to embedding.
        
        Uses a combination of:
        1. PCA-like basis (captures major coverage patterns)
        2. Spatial grouping (groups nearby sphere points)
        """
        # Method: Random orthogonal projection
        # In a learned version, this could be a neural network
        raw = np.random.randn(self.n_samples, self.embed_dim)
        # Normalize columns
        projection = raw / (np.linalg.norm(raw, axis=0, keepdims=True) + 1e-8)
        return projection.astype(np.float32)
    
    def compute_coverage_vector(
        self,
        viewing_directions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute coverage at each sphere sample point.
        
        Args:
            viewing_directions: List of unit vectors (camera->object directions)
            weights: Optional weight per observation (based on distance, resolution)
        
        Returns:
            Coverage vector of shape (n_samples,) with values in [0, 1]
        """
        if len(viewing_directions) == 0:
            return np.zeros(self.n_samples, dtype=np.float32)
        
        viewing_dirs = np.array(viewing_directions)  # (n_obs, 3)
        
        if weights is None:
            weights = np.ones(len(viewing_directions))
        weights = np.array(weights)
        
        # Compute angular distance from each sphere point to each observation
        # dot product: (n_samples, 3) @ (3, n_obs) -> (n_samples, n_obs)
        dots = np.clip(self.sphere_points @ viewing_dirs.T, -1.0, 1.0)
        angles = np.arccos(dots)  # (n_samples, n_obs)
        
        # Compute contribution using Gaussian-like decay
        contributions = weights[None, :] * np.exp(-(angles / self.fov_rad) ** self.decay_power)
        contributions = np.clip(contributions, 0.0, 1.0)
        
        # Soft union: 1 - product(1 - contribution)
        coverage = 1.0 - np.prod(1.0 - contributions, axis=1)
        
        return coverage.astype(np.float32)
    
    def encode(
        self,
        viewing_directions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Encode observation history into fixed-size embedding.
        
        Args:
            viewing_directions: List of viewing direction unit vectors
            weights: Optional weights per observation
        
        Returns:
            Embedding vector of shape (embed_dim,)
        """
        # Compute raw coverage on sphere
        coverage = self.compute_coverage_vector(viewing_directions, weights)
        
        # Project to embedding space
        embedding = coverage @ self.projection
        
        # Normalize to reasonable range
        embedding = np.tanh(embedding)
        
        return embedding.astype(np.float32)
    
    def get_coverage_stats(
        self,
        viewing_directions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Get interpretable coverage statistics.
        
        Returns:
            Dictionary with coverage metrics
        """
        coverage = self.compute_coverage_vector(viewing_directions, weights)
        
        return {
            'mean_coverage': float(np.mean(coverage)),
            'min_coverage': float(np.min(coverage)),
            'max_coverage': float(np.max(coverage)),
            'coverage_std': float(np.std(coverage)),
            'covered_ratio': float(np.mean(coverage > 0.5)),  # % of sphere well-covered
            'uncovered_ratio': float(np.mean(coverage < 0.1))  # % of sphere poorly covered
        }


@dataclass
class PendingUpload:
    """Represents an upload in progress."""
    camera_id: int
    resolution_id: int
    frames_remaining: int
    start_frame: int


class AoIEnv(gym.Env):
    """
    Gym environment for AoI minimization through camera/resolution selection.
    
    Observation Space (47 dimensions):
        - pose_aoi_normalized[5]: Normalized pose AoI for each object
        - style_aoi[5]: Style AoI for each object (already in [0,1])
        - cooldown_normalized[1]: Frames until channel is free (0-1)
        - time_remaining[1]: Episode progress (0-1)
        - camera_obs_counts[5]: Normalized observation count per camera
        - object_poses_6dof[30]: 5 objects × 6 DoF (normalized)
    
    Action Space (Discrete 10):
        - Actions 0-4: Camera 0-4 with Resolution 1 (640×360, 1 frame cooldown)
        - Actions 5-9: Camera 0-4 with Resolution 2 (1280×720, 4 frame cooldown)
    
    Reward:
        - r_t = -(alpha * sum(pose_aoi) + (1-alpha) * sum(style_aoi))
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        data_loader: Optional[PoseDataLoader] = None,
        variance_predictor: Optional[Any] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the AoI environment.
        
        Args:
            config: Configuration dictionary
            data_loader: PoseDataLoader instance
            variance_predictor: VariancePredictorWrapper instance (optional)
            render_mode: Render mode for visualization
        """
        super().__init__()
        
        # Default configuration
        if config is not None:
            self.config = config
            self._config_source = "passed directly"
        else:
            # Try to load config.yaml
            import os
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config', 'config.yaml'
            )
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                self._config_source = f"loaded from {config_path}"
            else:
                self.config = self._default_config()
                self._config_source = "using hardcoded defaults (config.yaml not found)"
        
        # Environment parameters
        self.num_cameras = self.config['env']['num_cameras']
        self.num_objects = self.config['env']['num_objects']
        self.episode_length = self.config['env']['episode_length']
        self.pose_aoi_increments = np.array(self.config['env']['pose_aoi_increment'])
        
        # Resolution settings - handle both int and string keys from YAML
        resolutions = self.config['env']['resolutions']
        # YAML might parse keys as int or string, so try both
        def get_resolution(res_id):
            if res_id in resolutions:
                return resolutions[res_id]
            elif str(res_id) in resolutions:
                return resolutions[str(res_id)]
            else:
                raise KeyError(f"Resolution {res_id} not found in config")
        
        res1 = get_resolution(1)
        res2 = get_resolution(2)
        
        self.resolution_cooldowns = {
            1: res1['cooldown'],
            2: res2['cooldown']
        }
        self.resolution_sizes = {
            1: (res1['width'], res1['height']),
            2: (res2['width'], res2['height'])
        }
        
        # Reward settings
        self.alpha = self.config['reward']['alpha']
        self.reward_scale = self.config['reward']['reward_scale']
        self.pose_aoi_max = self.config['reward']['pose_aoi_max_expected']
        self.pose_aoi_norm = self.config['reward'].get('pose_aoi_norm', 0.1)
        self.style_aoi_norm = self.config['reward'].get('style_aoi_norm', 5.0)
        
        # Data loader
        if data_loader is None:
            self.data_loader = PoseDataLoader(
                render_output_path=self.config['data']['render_output_path'],
                num_cameras=self.num_cameras,
                num_objects=self.num_objects,
                object_mapping=self.config['env'].get('object_mapping')
            )
        else:
            self.data_loader = data_loader
        
        # Variance predictor (optional - use mock if not provided)
        self.variance_predictor = variance_predictor
        self.use_mock_variance = variance_predictor is None
        
        # AoI manager
        self.aoi_manager = AoIManager(
            num_objects=self.num_objects,
            pose_aoi_increments=self.pose_aoi_increments.tolist(),
            initial_pose_aoi=self.config['env']['initial_pose_aoi'],
            initial_style_aoi=self.config['env']['initial_style_aoi'],
            style_error_params=self.config.get('style_error')
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)  # 5 cameras × 2 resolutions
        
        # Observation space dimensions:
        # - pose_aoi_normalized: 5
        # - style_aoi: 5
        # - cooldown_normalized: 1
        # - time_remaining: 1
        # - camera_obs_counts: 5
        # - object_poses_6dof: 30
        # - spherical_coverage_embedding: 5 objects × coverage_embed_dim
        # Total: 47 + 5 * coverage_embed_dim
        
        self.coverage_embed_dim = self.config.get('coverage_embed_dim', 16)
        obs_dim = 47 + self.num_objects * self.coverage_embed_dim  # 47 + 5*16 = 127
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Spherical coverage encoder
        self.coverage_encoder = SphericalCoverageEncoder(
            n_samples=64,
            embed_dim=self.coverage_embed_dim
        )
        
        # Internal state
        self.current_frame = 0
        self.cooldown_timer = 0
        self.pending_upload: Optional[PendingUpload] = None
        self.camera_obs_counts = np.zeros(self.num_cameras)
        
        # Viewpoint tracking per object: list of (viewing_direction, weight)
        self.object_viewpoints: List[List[Tuple[np.ndarray, float]]] = [[] for _ in range(self.num_objects)]
        
        # Augmentation settings
        training_config = self.config.get('training', {})
        self.augmentation = training_config.get('augmentation', {})
        
        # Render mode
        self.render_mode = render_mode
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_pose_aoi_history = []
        self.episode_style_aoi_history = []
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'env': {
                'num_cameras': 5,
                'num_objects': 5,
                'episode_length': 200,
                'pose_aoi_increment': [0.006, 0.02, 0.05, 0.01, 0.036],
                'initial_pose_aoi': 0.0,
                'initial_style_aoi': 1.0,
                'resolutions': {
                    1: {'cooldown': 1, 'width': 640, 'height': 360},
                    2: {'cooldown': 4, 'width': 1280, 'height': 720}
                },
                'object_mapping': {
                    'actor197': 0, 'actor198': 1, 'actor199': 2,
                    'actor200': 3, 'actor201': 4
                }
            },
            'reward': {
                'alpha': 0.5,
                'reward_scale': 1.0,
                'pose_aoi_max_expected': 10.0,
                'pose_aoi_norm': 0.1,    # Normalization factor for pose AoI
                'style_aoi_norm': 5.0    # Normalization factor for style AoI
            },
            'style_error': {
                'k': 4.0, 'p': 8.0, 'alpha': 0.2,
                'beta': 0.6, 'zeta': 1.0,
                'default_fov_deg': 45.0, 'n_samples': 1024
            },
            'data': {
                'render_output_path': './data/render_output',
                'variance_model_path': './models/variance_predictor_final.pth'
            },
            'training': {
                'augmentation': {
                    'random_start_frame': True,
                    'max_start_frame': 50,
                    'pose_noise_std': 0.001
                }
            }
        }
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset AoI manager
        self.aoi_manager.reset()
        
        # Reset frame counter (with optional random start)
        if self.augmentation.get('random_start_frame', False):
            max_start = min(
                self.augmentation.get('max_start_frame', 50),
                self.data_loader.num_frames - self.episode_length
            )
            max_start = max(0, max_start)
            self.start_frame = self.np_random.integers(0, max_start + 1)
        else:
            self.start_frame = 0
        
        self.current_frame = 0
        self.cooldown_timer = 0
        self.pending_upload = None
        self.camera_obs_counts = np.zeros(self.num_cameras)
        
        # Reset viewpoint tracking (list of viewing directions per object)
        self.object_viewpoints = [[] for _ in range(self.num_objects)]
        
        # Reset episode statistics
        self.episode_rewards = []
        self.episode_pose_aoi_history = []
        self.episode_style_aoi_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-9)
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information including:
                - decision: The action taken (camera_id, resolution_id)
                - pose_contribution: Weighted pose AoI contribution
                - style_contribution: Weighted style AoI contribution
        """
        # Record the decision
        decision = {
            'action': action,
            'camera_id': action % 5,
            'resolution_id': 1 if action < 5 else 2,
            'resolution_name': '640x360' if action < 5 else '1280x720',
            'action_taken': self.cooldown_timer == 0
        }
        
        # Step 1: Increment pose AoI for all objects
        self.aoi_manager.increment_pose_aoi()
        
        # Step 2: Check if pending upload completes
        observation_completed = False
        if self.pending_upload is not None:
            self.pending_upload.frames_remaining -= 1
            
            if self.pending_upload.frames_remaining == 0:
                # Upload completes - update AoIs
                self._process_observation_complete()
                observation_completed = True
                self.pending_upload = None
        
        # Step 3: Process new action (if cooldown is 0)
        if self.cooldown_timer == 0:
            camera_id = action % 5
            resolution_id = 1 if action < 5 else 2
            cooldown_frames = self.resolution_cooldowns[resolution_id]
            
            self.pending_upload = PendingUpload(
                camera_id=camera_id,
                resolution_id=resolution_id,
                frames_remaining=cooldown_frames,
                start_frame=self.current_frame
            )
            self.cooldown_timer = cooldown_frames
            self.camera_obs_counts[camera_id] += 1
        
        # Step 4: Update timers
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        
        self.current_frame += 1
        
        # Step 5: Compute reward
        reward = self._compute_reward()
        
        # Step 6: Check termination
        terminated = self.current_frame >= self.episode_length
        truncated = False
        
        # Record statistics
        self.episode_rewards.append(reward)
        self.episode_pose_aoi_history.append(self.aoi_manager.get_pose_aoi().copy())
        self.episode_style_aoi_history.append(self.aoi_manager.get_style_aoi().copy())
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add decision info
        info['decision'] = decision
        info['observation_completed'] = observation_completed
        info['reward'] = float(reward)
        
        return observation, reward, terminated, truncated, info
    
    def _process_observation_complete(self):
        """Process completed observation - update all object AoIs and viewpoint tracking."""
        if self.pending_upload is None:
            return
        
        camera_id = self.pending_upload.camera_id
        resolution_id = self.pending_upload.resolution_id
        width, height = self.resolution_sizes[resolution_id]
        
        # Get camera position
        camera_position = self.data_loader.get_camera_position(camera_id)
        
        # Get actual frame in data (accounting for start offset)
        data_frame = self.start_frame + self.current_frame + 1
        data_frame = min(data_frame, self.data_loader.num_frames)
        
        # Update AoI for all objects
        for obj_id in range(self.num_objects):
            # Get object pose
            pose = self.data_loader.get_pose(data_frame, camera_id, obj_id)
            object_position = pose[:3, 3]
            
            # Get variance from predictor
            variance_6dof = self._get_variance(pose, obj_id, camera_id, resolution_id)
            
            # Update pose AoI
            self.aoi_manager.update_pose_aoi(obj_id, variance_6dof)
            
            # Add observation record for style error
            self.aoi_manager.add_observation(
                object_id=obj_id,
                frame=self.current_frame,
                camera_id=camera_id,
                resolution_id=resolution_id,
                camera_position=camera_position,
                object_position=object_position,
                width=width,
                height=height,
                fov_deg=45.0
            )
            
            # Update style AoI
            self.aoi_manager.update_style_aoi(obj_id)
            
            # Track viewpoint for spherical coverage encoder
            viewing_direction = self._compute_viewing_direction(camera_position, object_position)
            distance = np.linalg.norm(object_position - camera_position)
            # Weight based on resolution and distance
            resolution_weight = 1.0 if resolution_id == 1 else 1.5  # Higher res = more weight
            distance_weight = 1.0 / (1.0 + 0.1 * distance ** 2)
            weight = resolution_weight * distance_weight
            self.object_viewpoints[obj_id].append((viewing_direction, weight))
    
    def _compute_viewing_direction(self, camera_pos: np.ndarray, object_pos: np.ndarray) -> np.ndarray:
        """
        Compute the unit viewing direction from camera to object.
        
        Returns:
            Unit vector (3,)
        """
        direction = object_pos - camera_pos
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            return direction / norm
        return np.array([0.0, 0.0, 1.0])
    
    def _get_variance(
        self,
        pose: np.ndarray,
        object_id: int,
        camera_id: int,
        resolution_id: int
    ) -> float:
        """
        Get 6-DoF variance from variance predictor.
        
        Args:
            pose: 4x4 pose matrix
            object_id: Object ID
            camera_id: Camera ID
            resolution_id: Resolution ID
        
        Returns:
            Combined 6-DoF variance (position + rotation)
        """
        if self.use_mock_variance:
            # Mock variance based on resolution
            base_variance = 0.01 if resolution_id == 2 else 0.02
            # Add some randomness
            noise = self.np_random.uniform(0.8, 1.2)
            return base_variance * noise
        
        # Use actual variance predictor
        result = self.variance_predictor.predict(
            prev_pose=pose,
            object_id=object_id,
            camera_id=camera_id,
            resolution_id=resolution_id
        )
        
        # Combine position and rotation variance
        # Using mean of position std + mean of rotation std
        pos_var = np.mean(result['position_std'])
        rot_var = np.mean(result['rotation_std'])
        
        return pos_var + rot_var
    
    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector (97 dimensions).
        
        Components:
        - pose_aoi_normalized (5): Current pose AoI per object
        - style_aoi (5): Current style AoI per object  
        - cooldown_normalized (1): Frames until channel free
        - time_remaining (1): Episode progress
        - camera_obs_counts (5): Total observations per camera
        - object_poses_6dof (30): 5 objects × 6 DoF
        - spherical_coverage_embedding (5 × embed_dim): Coverage embedding per object
        
        Returns:
            Observation array (47 + 5 * embed_dim,)
        """
        obs = []
        
        # 1. Pose AoI normalized (5 dims)
        pose_aoi = self.aoi_manager.get_pose_aoi()
        pose_aoi_norm = pose_aoi / self.pose_aoi_max
        obs.extend(pose_aoi_norm)
        
        # 2. Style AoI (5 dims) - already in [0, 1]
        style_aoi = self.aoi_manager.get_style_aoi()
        obs.extend(style_aoi)
        
        # 3. Cooldown normalized (1 dim)
        cooldown_norm = self.cooldown_timer / 4.0
        obs.append(cooldown_norm)
        
        # 4. Time remaining (1 dim)
        time_remaining = (self.episode_length - self.current_frame) / self.episode_length
        obs.append(time_remaining)
        
        # 5. Camera observation counts normalized (5 dims)
        total_obs = max(1, np.sum(self.camera_obs_counts))
        cam_obs_norm = self.camera_obs_counts / total_obs
        obs.extend(cam_obs_norm)
        
        # 6. Object poses 6-DoF (30 dims)
        data_frame = self.start_frame + self.current_frame + 1
        data_frame = min(data_frame, self.data_loader.num_frames)
        
        for obj_id in range(self.num_objects):
            pose_6dof = self.data_loader.get_pose_6dof(data_frame, 0, obj_id)
            
            # Normalize pose
            pose_6dof_norm = pose_6dof.copy()
            pose_6dof_norm[:3] /= 10.0
            pose_6dof_norm[3:] /= np.pi
            
            # Add noise if configured
            if self.augmentation.get('pose_noise_std', 0) > 0:
                noise = self.np_random.normal(
                    0, self.augmentation['pose_noise_std'], 6
                )
                pose_6dof_norm += noise
            
            obs.extend(pose_6dof_norm)
        
        # 7. Spherical coverage embedding per object (5 × embed_dim dims)
        for obj_id in range(self.num_objects):
            viewpoints = self.object_viewpoints[obj_id]
            if len(viewpoints) > 0:
                directions = [vp[0] for vp in viewpoints]
                weights = [vp[1] for vp in viewpoints]
                embedding = self.coverage_encoder.encode(directions, weights)
            else:
                # No observations yet - zero embedding
                embedding = np.zeros(self.coverage_embed_dim, dtype=np.float32)
            obs.extend(embedding)
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self) -> float:
        """
        Compute the reward for current state.
        
        Uses normalized AoI values so alpha controls actual proportion:
        reward = -(alpha * pose_aoi/pose_norm + (1-alpha) * style_aoi/style_norm)
        
        Returns:
            Reward value
        """
        total_pose = np.sum(self.aoi_manager.get_pose_aoi())
        total_style = np.sum(self.aoi_manager.get_style_aoi())
        
        # Normalize to similar scales
        pose_normalized = total_pose / self.pose_aoi_norm
        style_normalized = total_style / self.style_aoi_norm
        
        # Weighted sum
        total_aoi = self.alpha * pose_normalized + (1 - self.alpha) * style_normalized
        reward = -total_aoi * self.reward_scale
        return reward
    
    def _get_info(self) -> Dict:
        """
        Get additional information about current state.
        
        Returns:
            Info dictionary
        """
        pose_aoi = self.aoi_manager.get_pose_aoi()
        style_aoi = self.aoi_manager.get_style_aoi()
        total_pose = np.sum(pose_aoi)
        total_style = np.sum(style_aoi)
        
        # Normalized contributions (so alpha controls the actual proportion)
        pose_normalized = total_pose / self.pose_aoi_norm
        style_normalized = total_style / self.style_aoi_norm
        
        pose_contribution = self.alpha * pose_normalized
        style_contribution = (1 - self.alpha) * style_normalized
        total_aoi = pose_contribution + style_contribution
        
        # Last action info
        last_action_info = None
        if self.pending_upload is not None:
            last_action_info = {
                'camera_id': self.pending_upload.camera_id,
                'resolution_id': self.pending_upload.resolution_id,
                'frames_remaining': self.pending_upload.frames_remaining
            }
        
        return {
            # Frame info
            'frame': self.current_frame,
            'cooldown': self.cooldown_timer,
            'can_act': self.cooldown_timer == 0,
            
            # Per-object AoI (raw values)
            'pose_aoi': pose_aoi.tolist(),
            'style_aoi': style_aoi.tolist(),
            
            # Totals (raw values)
            'total_pose_aoi': float(total_pose),
            'total_style_aoi': float(total_style),
            
            # Normalization factors
            'pose_aoi_norm': self.pose_aoi_norm,
            'style_aoi_norm': self.style_aoi_norm,
            
            # Normalized values
            'pose_normalized': float(pose_normalized),
            'style_normalized': float(style_normalized),
            
            # Weighted contributions (for reward)
            'alpha': self.alpha,
            'pose_contribution': float(pose_contribution),
            'style_contribution': float(style_contribution),
            'total_aoi': float(total_aoi),
            
            # Contribution percentages
            'pose_contribution_pct': float(pose_contribution / total_aoi * 100) if total_aoi > 0 else 0.0,
            'style_contribution_pct': float(style_contribution / total_aoi * 100) if total_aoi > 0 else 0.0,
            
            # Action info
            'pending_upload': last_action_info,
            'camera_obs_counts': self.camera_obs_counts.tolist(),
            
            # Spherical coverage stats per object
            'coverage_stats': self._compute_coverage_stats()
        }
    
    def _compute_coverage_stats(self) -> Dict:
        """
        Compute spherical coverage statistics for each object.
        
        Returns:
            Dictionary with coverage metrics per object
        """
        stats = {}
        
        for obj_id in range(self.num_objects):
            viewpoints = self.object_viewpoints[obj_id]
            if len(viewpoints) > 0:
                directions = [vp[0] for vp in viewpoints]
                weights = [vp[1] for vp in viewpoints]
                obj_stats = self.coverage_encoder.get_coverage_stats(directions, weights)
                obj_stats['num_observations'] = len(viewpoints)
            else:
                obj_stats = {
                    'mean_coverage': 0.0,
                    'min_coverage': 0.0,
                    'max_coverage': 0.0,
                    'coverage_std': 0.0,
                    'covered_ratio': 0.0,
                    'uncovered_ratio': 1.0,
                    'num_observations': 0
                }
            stats[f'object_{obj_id}'] = obj_stats
        
        # Overall stats
        total_mean_coverage = np.mean([
            stats[f'object_{i}']['mean_coverage'] 
            for i in range(self.num_objects)
        ])
        total_observations = sum([
            stats[f'object_{i}']['num_observations'] 
            for i in range(self.num_objects)
        ])
        
        stats['overall'] = {
            'mean_coverage': float(total_mean_coverage),
            'total_observations': total_observations
        }
        
        return stats
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for valid actions.
        
        Returns:
            Boolean array of shape (10,) - True if action is valid
        """
        if self.cooldown_timer > 0:
            # No actions valid during cooldown
            return np.zeros(10, dtype=bool)
        else:
            # All actions valid
            return np.ones(10, dtype=bool)
    
    def set_alpha(self, alpha: float):
        """
        Set the alpha parameter for reward computation.
        
        Args:
            alpha: New alpha value (0-1)
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)
    
    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            # Print current state
            pose_aoi = self.aoi_manager.get_pose_aoi()
            style_aoi = self.aoi_manager.get_style_aoi()
            
            print(f"Frame {self.current_frame}/{self.episode_length}")
            print(f"  Cooldown: {self.cooldown_timer}")
            print(f"  Pose AoI: {pose_aoi}")
            print(f"  Style AoI: {style_aoi}")
            print(f"  Total AoI: {self.aoi_manager.get_total_aoi(self.alpha):.4f}")
    
    def close(self):
        """Clean up resources."""
        pass


# Wrapper for action masking with Stable-Baselines3
class ActionMaskedAoIEnv(AoIEnv):
    """
    AoI Environment with action masking support for SB3.
    
    During cooldown, forces a no-op by returning the same state.
    """
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with action masking."""
        # If in cooldown, ignore action (but still process frame)
        if self.cooldown_timer > 0:
            # Still need to process the frame (increment AoI, check pending)
            return self._step_no_action()
        else:
            return super().step(action)
    
    def _step_no_action(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Process a frame without taking a new action."""
        # Step 1: Increment pose AoI
        self.aoi_manager.increment_pose_aoi()
        
        # Step 2: Check pending upload
        if self.pending_upload is not None:
            self.pending_upload.frames_remaining -= 1
            if self.pending_upload.frames_remaining == 0:
                self._process_observation_complete()
                self.pending_upload = None
        
        # Step 3: Update timers
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        self.current_frame += 1
        
        # Step 4: Compute reward
        reward = self._compute_reward()
        
        # Step 5: Check termination
        terminated = self.current_frame >= self.episode_length
        truncated = False
        
        # Record statistics
        self.episode_rewards.append(reward)
        self.episode_pose_aoi_history.append(self.aoi_manager.get_pose_aoi().copy())
        self.episode_style_aoi_history.append(self.aoi_manager.get_style_aoi().copy())
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info


def test_environment():
    """Test the AoI environment."""
    print("Testing AoI Environment")
    print("=" * 50)
    
    env = AoIEnv()
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps with random actions
    print("\nRunning 20 steps with random actions...")
    total_reward = 0
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 5 == 0:
            print(f"Step {i}: action={action}, reward={reward:.4f}, "
                  f"total_aoi={info['total_aoi']:.4f}, can_act={info['can_act']}")
    
    print(f"\nTotal reward over 20 steps: {total_reward:.4f}")
    print(f"Final pose AoI: {info['pose_aoi']}")
    print(f"Final style AoI: {info['style_aoi']}")
    
    # Test action masking
    print("\nTesting action mask:")
    mask = env.get_action_mask()
    print(f"Action mask: {mask}")
    
    env.close()
    print("\nTest completed!")


if __name__ == "__main__":
    test_environment()
