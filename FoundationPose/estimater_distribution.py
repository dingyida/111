# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm


class PoseDistribution:
    """
  Represents a Gaussian distribution over SE(3) poses.

  The distribution is parameterized in the tangent space:
  - Mean: 4x4 transformation matrix
  - Covariance: 6x6 matrix in tangent space [translation(3), rotation_log(3)]
  """

    def __init__(self, mean_pose, covariance, pose_samples=None, weights=None):
        """
    Args:
        mean_pose: (4,4) numpy array - mean transformation matrix
        covariance: (6,6) numpy array - covariance in tangent space
        pose_samples: Optional (N,4,4) - the discrete pose samples used to compute this
        weights: Optional (N,) - the weights/scores for each sample
    """
        self.mean_pose = mean_pose
        self.covariance = covariance
        self.pose_samples = pose_samples
        self.weights = weights

    def get_position_std(self):
        """Get standard deviation of position (3D vector)"""
        return np.sqrt(np.diag(self.covariance[:3, :3]))

    def get_rotation_std(self):
        """Get standard deviation of rotation in axis-angle (3D vector, in radians)"""
        return np.sqrt(np.diag(self.covariance[3:, 3:]))

    def get_confidence_ellipsoid(self, n_std=2):
        """
    Get confidence ellipsoid parameters for visualization.
    Returns eigenvalues and eigenvectors for position uncertainty.
    """
        pos_cov = self.covariance[:3, :3]
        eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)
        # Scale by n_std for n-sigma ellipsoid
        radii = n_std * np.sqrt(eigenvalues)
        return radii, eigenvectors

    def sample(self, n_samples=100):
        """
    Sample poses from the Gaussian distribution.

    Args:
        n_samples: Number of samples to generate

    Returns:
        (n_samples, 4, 4) array of transformation matrices
    """
        # Sample from 6D tangent space
        tangent_samples = np.random.multivariate_normal(
            mean=np.zeros(6),
            cov=self.covariance,
            size=n_samples
        )

        # Convert back to SE(3)
        pose_samples = np.zeros((n_samples, 4, 4))
        for i in range(n_samples):
            pose_samples[i] = self._tangent_to_pose(tangent_samples[i], self.mean_pose)

        return pose_samples

    @staticmethod
    def _tangent_to_pose(tangent_vec, reference_pose):
        """Convert tangent space vector to SE(3) pose relative to reference"""
        delta_trans = tangent_vec[:3]
        delta_rot_vec = tangent_vec[3:]

        # Convert rotation vector to matrix
        delta_rot = R.from_rotvec(delta_rot_vec).as_matrix()

        # Compose with reference pose
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = delta_rot
        delta_pose[:3, 3] = delta_trans

        return reference_pose @ delta_pose


def pose_to_tangent_space(pose, reference_pose):
    """
  Convert SE(3) pose to 6D tangent space representation relative to reference.

  Args:
      pose: (4,4) transformation matrix
      reference_pose: (4,4) reference transformation matrix

  Returns:
      (6,) vector [translation(3), rotation_log(3)]
  """
    # Compute relative transformation
    relative_pose = np.linalg.inv(reference_pose) @ pose

    # Extract translation (straightforward)
    translation = relative_pose[:3, 3]

    # Extract rotation and convert to log map (axis-angle)
    rot_matrix = relative_pose[:3, :3]
    rotation_vec = R.from_matrix(rot_matrix).as_rotvec()

    return np.concatenate([translation, rotation_vec])


def compute_pose_mean_on_manifold(poses, weights=None):
    """
  Compute weighted Fr¨¦chet mean of poses on SE(3) manifold.

  Args:
      poses: (N, 4, 4) array of transformation matrices
      weights: (N,) array of weights (will be normalized)

  Returns:
      (4, 4) mean transformation matrix
  """
    N = len(poses)

    if weights is None:
        weights = np.ones(N) / N
    else:
        weights = weights / weights.sum()

    # Initialize with weighted average of translations
    mean_translation = np.sum(poses[:, :3, 3] * weights[:, None], axis=0)

    # Compute Fr¨¦chet mean of rotations using iterative algorithm
    rotations = [R.from_matrix(pose[:3, :3]) for pose in poses]

    # Start with first rotation as initial guess
    mean_rotation = rotations[0]

    # Iterative refinement (gradient descent on manifold)
    for _ in range(50):  # Max iterations
        # Compute log maps of all rotations relative to current mean
        tangent_vecs = []
        for i, rot in enumerate(rotations):
            relative_rot = mean_rotation.inv() * rot
            tangent_vec = relative_rot.as_rotvec()
            tangent_vecs.append(tangent_vec * weights[i])

        # Weighted mean in tangent space
        mean_tangent = np.sum(tangent_vecs, axis=0)

        # Check convergence
        if np.linalg.norm(mean_tangent) < 1e-6:
            break

        # Update mean by exponential map
        update = R.from_rotvec(mean_tangent)
        mean_rotation = mean_rotation * update

    # Construct mean pose
    mean_pose = np.eye(4)
    mean_pose[:3, :3] = mean_rotation.as_matrix()
    mean_pose[:3, 3] = mean_translation

    return mean_pose


def fit_gaussian_to_poses(poses, scores, top_k=None):
    """
  Fit a Gaussian distribution to a set of weighted pose samples using MLE.

  Args:
      poses: (N, 4, 4) array of transformation matrices
      scores: (N,) array of confidence scores
      top_k: Optional, use only top-k poses

  Returns:
      PoseDistribution object
  """
    N = len(poses)

    if top_k is not None and top_k < N:
        # Select top-k by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        poses = poses[top_indices]
        scores = scores[top_indices]
        N = len(poses)  # Update N after filtering

    # Normalize scores to get weights
    weights = scores / scores.sum()

    # Compute mean pose on manifold
    mean_pose = compute_pose_mean_on_manifold(poses, weights)

    # Convert all poses to tangent space relative to mean
    tangent_vectors = np.array([
        pose_to_tangent_space(pose, mean_pose) for pose in poses
    ])

    # Compute weighted covariance in tangent space
    weighted_mean_tangent = np.sum(tangent_vectors * weights[:, None], axis=0)

    centered = tangent_vectors - weighted_mean_tangent
    covariance = np.zeros((6, 6))
    for i in range(N):
        covariance += weights[i] * np.outer(centered[i], centered[i])

    # Add small regularization for numerical stability
    covariance += np.eye(6) * 1e-8

    return PoseDistribution(
        mean_pose=mean_pose,
        covariance=covariance,
        pose_samples=poses,
        weights=weights
    )


class FoundationPose:
    def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer: ScorePredictor = None,
                 refiner: PoseRefinePredictor = None, glctx=None, debug=0,
                 debug_dir='/home/bowen/debug/novel_pose_debug/', distribution_top_k=20):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir
        self.distribution_top_k = distribution_top_k  # Number of top poses to use for distribution
        os.makedirs(debug_dir, exist_ok=True)

        self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = glctx

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = ScorePredictor()

        if refiner is not None:
            self.refiner = refiner
        else:
            self.refiner = PoseRefinePredictor()

        self.pose_last = None  # Used for tracking; per the centered mesh
        self.pose_distribution_last = None  # Distribution from last frame

    def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2
        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

        model_pts = mesh.vertices
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg
        pcd = toOpen3dCloud(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
        logging.info(f'self.pts:{self.pts.shape}')
        self.mesh_path = None
        self.mesh = mesh
        if self.mesh is not None:
            self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
            self.mesh.export(self.mesh_path)
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

        logging.info("reset done")

    def get_tf_to_centered_mesh(self):
        tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
        return tf_to_center

    def to_device(self, s='cuda:0'):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k]
            if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
                logging.info(f"Moving {k} to device {s}")
                self.__dict__[k] = self.__dict__[k].to(s)
        for k in self.mesh_tensors:
            logging.info(f"Moving {k} to device {s}")
            self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        if self.refiner is not None:
            self.refiner.model.to(s)
        if self.scorer is not None:
            self.scorer.model.to(s)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(s)

    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        logging.info(f'cam_in_obs:{cam_in_obs.shape}')
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        logging.info(f"rot_grid:{rot_grid.shape}")
        rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
        self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
        logging.info(f"self.rot_grid: {self.rot_grid.shape}")

    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
        '''
    @scene_pts: torch tensor (N,3)
    '''
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1, 3)
        return ob_in_cams

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            logging.info(f'mask is all zero')
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            logging.info(f"valid is empty")
            return np.zeros((3))

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        if self.debug >= 2:
            pcd = toOpen3dCloud(center.reshape(1, 3))
            o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

        return center.reshape(3)

    def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, return_distribution=True):
        '''
    Compute pose distribution from given pts to self.pcd

    Args:
        K: Camera intrinsics
        rgb: RGB image
        depth: Depth map
        ob_mask: Object mask
        ob_id: Object ID (optional)
        glctx: Graphics context (optional)
        iteration: Number of refinement iterations
        return_distribution: If True, return PoseDistribution object; else return best pose only

    Returns:
        If return_distribution=True: (best_pose, PoseDistribution)
        If return_distribution=False: best_pose (for backward compatibility)
    '''
        set_seed(0)
        logging.info('Welcome')

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
                # self.glctx = dr.RasterizeGLContext()
            else:
                self.glctx = glctx

        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')

        if self.debug >= 2:
            xyz_map = depth2xyzmap(depth, K)
            valid = xyz_map[..., 2] >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply', pcd)
            cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask * 255.0).clip(0, 255))

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            logging.info(f'valid too small, return')
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)

            if return_distribution:
                # Return with high uncertainty
                high_uncertainty_cov = np.eye(6) * 1.0  # Large uncertainty
                dist = PoseDistribution(pose, high_uncertainty_cov)
                return pose, dist
            else:
                return pose

        if self.debug >= 2:
            imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
            cv2.imwrite(f'{self.debug_dir}/depth.png', (depth * 1000).astype(np.uint16))
            valid = xyz_map[..., 2] >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
        poses = poses.data.cpu().numpy()
        logging.info(f'poses:{poses.shape}')
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')

        add_errs = self.compute_add_err_to_gt_pose(poses)
        logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

        xyz_map = depth2xyzmap(depth, K)
        poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                          glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

        add_errs = self.compute_add_err_to_gt_pose(poses)
        logging.info(f"final, add_errs min:{add_errs.min()}")

        # Convert scores to numpy if needed, then sort
        if torch.is_tensor(scores):
            scores_np = scores.cpu().numpy()
        elif isinstance(scores, np.ndarray):
            scores_np = scores
        else:
            scores_np = np.array(scores)
        ids = np.argsort(scores_np)[::-1]  # Descending order
        logging.info(f'sort ids:{ids}')
        scores = scores_np[ids]

        # Handle poses indexing - convert if needed to avoid negative stride issue
        if torch.is_tensor(poses):
            # Convert ids to torch tensor for indexing, or convert poses to numpy
            poses_np = poses.data.cpu().numpy() if poses.is_cuda else poses.numpy()
            poses = poses_np[ids]
        else:
            poses = poses[ids]

        logging.info(f'sorted scores:{scores}')

        # Convert tensor to numpy for matrix multiplication
        tf_to_centered = self.get_tf_to_centered_mesh()
        tf_to_centered_np = tf_to_centered.cpu().numpy() if torch.is_tensor(tf_to_centered) else tf_to_centered
        best_pose = poses[0] @ tf_to_centered_np
        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores

        # Fit Gaussian distribution to top-K poses
        if return_distribution:
            poses_np = poses.data.cpu().numpy() if torch.is_tensor(poses) else poses
            scores_np = scores  # Already numpy from above

            pose_dist = fit_gaussian_to_poses(
                poses_np,
                scores_np,
                top_k=self.distribution_top_k
            )

            # Transform distribution to be relative to original mesh (not centered)
            tf_to_centered = self.get_tf_to_centered_mesh().data.cpu().numpy()
            pose_dist.mean_pose = pose_dist.mean_pose @ tf_to_centered

            # Transform samples as well if they exist
            if pose_dist.pose_samples is not None:
                pose_dist.pose_samples = np.array([
                    p @ tf_to_centered for p in pose_dist.pose_samples
                ])

            self.pose_distribution_last = pose_dist

            logging.info(f"Pose distribution - Position std: {pose_dist.get_position_std()}")
            logging.info(f"Pose distribution - Rotation std (deg): {np.rad2deg(pose_dist.get_rotation_std())}")

            return best_pose, pose_dist
        else:
            return best_pose

    def compute_add_err_to_gt_pose(self, poses):
        '''
    @poses: wrt. the centered mesh
    '''
        return -torch.ones(len(poses), device='cuda', dtype=torch.float)

    def track_one(self, rgb, depth, K, iteration, return_distribution=True, extra={}):
        """
    Track pose in current frame.

    Args:
        rgb: RGB image
        depth: Depth map
        K: Camera intrinsics
        iteration: Number of refinement iterations
        return_distribution: If True, return PoseDistribution; else return best pose only
        extra: Extra information dict

    Returns:
        If return_distribution=True: (best_pose, PoseDistribution)
        If return_distribution=False: best_pose (for backward compatibility)
    """
        if self.pose_last is None:
            logging.info("Please init pose by register first")
            raise RuntimeError
        logging.info("Welcome")

        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        logging.info("depth processing done")

        xyz_map = \
        depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

        # For tracking, we can either:
        # 1. Just refine the last best pose (faster, less uncertainty info)
        # 2. Propagate multiple hypotheses from last distribution (more robust, better uncertainty)

        if return_distribution and self.pose_distribution_last is not None:
            # Strategy: Sample from last distribution or use top-K from last frame
            # Here we'll use the stored pose samples from last frame
            num_hypotheses = min(self.distribution_top_k, len(self.poses))

            # Take top-K poses from last frame as starting points
            initial_poses = self.poses[:num_hypotheses] if isinstance(self.poses, np.ndarray) else self.poses[:num_hypotheses].data.cpu().numpy()

            # Refine each hypothesis
            refined_poses, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=initial_poses,
                normal_map=None,
                xyz_map=xyz_map,
                mesh_diameter=self.diameter,
                glctx=self.glctx,
                iteration=iteration,
                get_vis=self.debug >= 2
            )

            # Re-score the refined poses
            refined_scores, score_vis = self.scorer.predict(
                mesh=self.mesh,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=refined_poses.data.cpu().numpy() if torch.is_tensor(refined_poses) else refined_poses,
                normal_map=None,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                get_vis=self.debug >= 2
            )

            if self.debug >= 2 and vis is not None:
                extra['vis'] = vis

            # Sort by score - ensure scores are numpy
            if torch.is_tensor(refined_scores):
                refined_scores_np = refined_scores.cpu().numpy()
            elif isinstance(refined_scores, np.ndarray):
                refined_scores_np = refined_scores
            else:
                refined_scores_np = np.array(refined_scores)
            ids = np.argsort(refined_scores_np)[::-1]

            # Handle poses indexing - convert if needed to avoid negative stride issue
            if torch.is_tensor(refined_poses):
                refined_poses_np = refined_poses.data.cpu().numpy() if refined_poses.is_cuda else refined_poses.numpy()
                refined_poses_sorted = refined_poses_np[ids]
            else:
                refined_poses_sorted = refined_poses[ids]

            refined_scores_sorted = refined_scores_np[ids]

            # Update state
            self.pose_last = refined_poses_sorted[0]
            self.poses = torch.as_tensor(refined_poses_sorted, device='cuda', dtype=torch.float)
            self.scores = refined_scores_sorted

            # Fit new distribution
            pose_dist = fit_gaussian_to_poses(
                refined_poses_sorted,
                refined_scores_sorted,
                top_k=self.distribution_top_k
            )

            # Transform to original mesh coordinates
            tf_to_centered = self.get_tf_to_centered_mesh().data.cpu().numpy()
            pose_dist.mean_pose = pose_dist.mean_pose @ tf_to_centered
            if pose_dist.pose_samples is not None:
                pose_dist.pose_samples = np.array([
                    p @ tf_to_centered for p in pose_dist.pose_samples
                ])

            self.pose_distribution_last = pose_dist

            # Convert to tensor for multiplication, then back to numpy
            tf_to_centered_np = self.get_tf_to_centered_mesh().cpu().numpy()
            best_pose = (self.pose_last @ tf_to_centered_np).reshape(4, 4)

            logging.info(f"Track - Position std: {pose_dist.get_position_std()}")
            logging.info(f"Track - Rotation std (deg): {np.rad2deg(pose_dist.get_rotation_std())}")

            return best_pose, pose_dist

        else:
            # Simple tracking - refine single pose
            pose_last_np = self.pose_last if isinstance(self.pose_last, np.ndarray) else self.pose_last.data.cpu().numpy()
            pose, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=pose_last_np.reshape(1, 4, 4),
                normal_map=None,
                xyz_map=xyz_map,
                mesh_diameter=self.diameter,
                glctx=self.glctx,
                iteration=iteration,
                get_vis=self.debug >= 2
            )
            logging.info("pose done")
            if self.debug >= 2:
                extra['vis'] = vis

            self.pose_last = pose
            tf_to_centered_np = self.get_tf_to_centered_mesh().cpu().numpy()
            best_pose = (pose @ tf_to_centered_np).reshape(4, 4)

            if return_distribution:
                # Create distribution with high uncertainty since we only have one pose
                high_uncertainty_cov = np.eye(6) * 0.01
                pose_dist = PoseDistribution(best_pose, high_uncertainty_cov)
                return best_pose, pose_dist
            else:
                return best_pose
