# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater_distribution import *
from datareader import *
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_pose_distribution(pose_dist, save_path=None):
    """
    Visualize the pose distribution statistics.
    
    Args:
        pose_dist: PoseDistribution object
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position uncertainty
    pos_std = pose_dist.get_position_std()
    axes[0, 0].bar(['X', 'Y', 'Z'], pos_std * 1000)  # Convert to mm
    axes[0, 0].set_ylabel('Standard Deviation (mm)')
    axes[0, 0].set_title('Position Uncertainty')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rotation uncertainty
    rot_std = np.rad2deg(pose_dist.get_rotation_std())
    axes[0, 1].bar(['Roll', 'Pitch', 'Yaw'], rot_std)
    axes[0, 1].set_ylabel('Standard Deviation (degrees)')
    axes[0, 1].set_title('Rotation Uncertainty')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Covariance matrix heatmap - Position
    pos_cov = pose_dist.covariance[:3, :3]
    im1 = axes[1, 0].imshow(pos_cov * 1e6, cmap='viridis', aspect='auto')  # Scale for visibility
    axes[1, 0].set_xticks([0, 1, 2])
    axes[1, 0].set_yticks([0, 1, 2])
    axes[1, 0].set_xticklabels(['X', 'Y', 'Z'])
    axes[1, 0].set_yticklabels(['X', 'Y', 'Z'])
    axes[1, 0].set_title('Position Covariance (¡Á10? m2)')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Covariance matrix heatmap - Rotation
    rot_cov = pose_dist.covariance[3:, 3:]
    im2 = axes[1, 1].imshow(rot_cov, cmap='viridis', aspect='auto')
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_yticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['Roll', 'Pitch', 'Yaw'])
    axes[1, 1].set_yticklabels(['Roll', 'Pitch', 'Yaw'])
    axes[1, 1].set_title('Rotation Covariance (rad2)')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_confidence_ellipsoid(pose_dist, K, img, bbox, save_path=None, n_std=2):
    """
    Visualize 3D confidence ellipsoid projected onto image.
    
    Args:
        pose_dist: PoseDistribution object
        K: Camera intrinsics
        img: Input image
        bbox: Bounding box for the object
        save_path: Optional path to save visualization
        n_std: Number of standard deviations for ellipsoid
    """
    # Get ellipsoid parameters
    radii, axes = pose_dist.get_confidence_ellipsoid(n_std=n_std)
    
    # Sample points on ellipsoid
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Rotate ellipsoid
    ellipsoid_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    rotated_points = ellipsoid_points @ axes.T
    
    # Translate to mean position
    mean_position = pose_dist.mean_pose[:3, 3]
    rotated_points += mean_position
    
    # Project to image
    points_homogeneous = np.hstack([rotated_points, np.ones((len(rotated_points), 1))])
    projected = (K @ points_homogeneous[:, :3].T).T
    projected_2d = projected[:, :2] / projected[:, 2:3]
    
    # Draw on image
    vis = img.copy()
    for i in range(len(projected_2d)):
        pt = tuple(projected_2d[i].astype(int))
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(vis, pt, 2, (0, 255, 255), -1)
    
    if save_path:
        cv2.imwrite(save_path, vis[..., ::-1])
    
    return vis


def save_pose_distribution(pose_dist, filepath):
    """
    Save pose distribution to file.
    
    Args:
        pose_dist: PoseDistribution object
        filepath: Path to save (e.g., .npz file)
    """
    np.savez(
        filepath,
        mean_pose=pose_dist.mean_pose,
        covariance=pose_dist.covariance,
        pose_samples=pose_dist.pose_samples if pose_dist.pose_samples is not None else np.array([]),
        weights=pose_dist.weights if pose_dist.weights is not None else np.array([])
    )


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/kinect_driller_seq/mesh/textured_mesh.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/kinect_driller_seq')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--distribution_top_k', type=int, default=20, help='Number of top poses to use for distribution fitting')
  parser.add_argument('--visualize_uncertainty', action='store_true', help='Generate uncertainty visualizations')
  parser.add_argument('--save_distributions', action='store_true', help='Save pose distributions to files')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam {debug_dir}/distributions {debug_dir}/uncertainty_vis')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
      model_pts=mesh.vertices, 
      model_normals=mesh.vertex_normals, 
      mesh=mesh, 
      scorer=scorer, 
      refiner=refiner, 
      debug_dir=debug_dir, 
      debug=debug, 
      glctx=glctx,
      distribution_top_k=args.distribution_top_k
  )
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  # Statistics tracking
  position_uncertainties = []
  rotation_uncertainties = []

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose, pose_dist = est.register(
          K=reader.K, 
          rgb=color, 
          depth=depth, 
          ob_mask=mask, 
          iteration=args.est_refine_iter,
          return_distribution=True
      )

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose, pose_dist = est.track_one(
          rgb=color, 
          depth=depth, 
          K=reader.K, 
          iteration=args.track_refine_iter,
          return_distribution=True
      )

    # Save pose (mean of distribution)
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    
    # Save distribution if requested
    if args.save_distributions:
      save_pose_distribution(
          pose_dist, 
          f'{debug_dir}/distributions/{reader.id_strs[i]}_dist.npz'
      )
    
    # Track uncertainty statistics
    position_uncertainties.append(pose_dist.get_position_std())
    rotation_uncertainties.append(pose_dist.get_rotation_std())
    
    # Log uncertainty
    logging.info(f"Frame {i} - Position std (mm): {pose_dist.get_position_std() * 1000}")
    logging.info(f"Frame {i} - Rotation std (deg): {np.rad2deg(pose_dist.get_rotation_std())}")

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      
      # Visualize uncertainty ellipsoid if requested
      if args.visualize_uncertainty:
        vis_ellipsoid = visualize_confidence_ellipsoid(
            pose_dist, 
            reader.K, 
            vis, 
            bbox,
            save_path=f'{debug_dir}/uncertainty_vis/{reader.id_strs[i]}_ellipsoid.png'
        )

    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    # Generate uncertainty statistics plot periodically
    if args.visualize_uncertainty and (i % 10 == 0 or i == len(reader.color_files) - 1):
      visualize_pose_distribution(
          pose_dist,
          save_path=f'{debug_dir}/uncertainty_vis/{reader.id_strs[i]}_stats.png'
      )

  # Generate summary statistics
  logging.info("\n========== Summary Statistics ==========")
  position_uncertainties = np.array(position_uncertainties)
  rotation_uncertainties = np.array(rotation_uncertainties)
  
  logging.info(f"Average position std (mm): {position_uncertainties.mean(axis=0) * 1000}")
  logging.info(f"Average rotation std (deg): {np.rad2deg(rotation_uncertainties.mean(axis=0))}")
  logging.info(f"Max position std (mm): {position_uncertainties.max(axis=0) * 1000}")
  logging.info(f"Max rotation std (deg): {np.rad2deg(rotation_uncertainties.max(axis=0))}")
  
  # Save summary
  if args.save_distributions:
    np.savez(
        f'{debug_dir}/distributions/summary_statistics.npz',
        position_uncertainties=position_uncertainties,
        rotation_uncertainties=rotation_uncertainties
    )
  
  # Plot uncertainty over time
  if args.visualize_uncertainty:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Position uncertainty over time
    axes[0].plot(position_uncertainties * 1000)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Position Std (mm)')
    axes[0].set_title('Position Uncertainty Over Time')
    axes[0].legend(['X', 'Y', 'Z'])
    axes[0].grid(True, alpha=0.3)
    
    # Rotation uncertainty over time
    axes[1].plot(np.rad2deg(rotation_uncertainties))
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Rotation Std (degrees)')
    axes[1].set_title('Rotation Uncertainty Over Time')
    axes[1].legend(['Roll', 'Pitch', 'Yaw'])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{debug_dir}/uncertainty_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
  
  logging.info(f"\nResults saved to {debug_dir}")
