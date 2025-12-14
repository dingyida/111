#!/usr/bin/env python3
"""
Visualize annotated 4x4 poses for multiple actors.

Reads:
  - RGB:  /home/yid324/FoundationPose/demo_data/object/rgb/000001.png ...
  - Poses:/home/yid324/FoundationPose/demo_data/object/annotated_poses/000001_actor197.txt ...

Draws a 3D bounding box + XYZ axes for each actor on top of the RGB image.
"""

import os
import numpy as np
import imageio
import trimesh
import logging

from estimater import *          # for draw_posed_3d_box, draw_xyz_axis, etc.
from datareader import *         # for YcbineoatReader


def main():
    # ---------- PATH CONFIG ----------
    root_dir   = "/home/yid324/FoundationPose/demo_data/object"
    rgb_root   = root_dir                     # YcbineoatReader expects the scene dir
    pose_root  = os.path.join(root_dir, "annotated_poses")
    mesh_file  = os.path.join(root_dir, "mesh", "textured_mesh.obj")
    out_dir    = os.path.join(root_dir, "annot_vis")

    os.makedirs(out_dir, exist_ok=True)

    # Actors you have poses for
    actor_ids = [197, 198, 199, 200, 201]

    # ---------- LOGGING ----------
    set_logging_format()
    set_seed(0)
    logging.info("Starting visualization from annotated poses")

    # ---------- LOAD MESH & COMPUTE BBOX ----------
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        print("[INFO] mesh_file loaded as a trimesh.Scene; merging all geometries.")
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # Oriented bounding box of the mesh
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ---------- READER FOR RGB + CAMERA K ----------
    reader = YcbineoatReader(video_dir=rgb_root, shorter_side=None, zfar=np.inf)
    logging.info(f"Loaded {len(reader.color_files)} RGB frames")

    # ---------- MAIN LOOP ----------
    for i in range(len(reader.color_files)):
        frame_id = reader.id_strs[i]  # e.g. "000001"
        logging.info(f"Processing frame {i} ({frame_id})")

        color = reader.get_color(i)   # H x W x 3, RGB
        vis = color.copy()

        for actor_id in actor_ids:
            pose_path = os.path.join(pose_root, f"{frame_id}_actor{actor_id}.txt")
            if not os.path.exists(pose_path):
                # No pose for this actor in this frame
                continue

            # 4x4 pose matrix: object -> camera
            pose = np.loadtxt(pose_path).reshape(4, 4)

            # Convert to OBB-centered frame for visualization
            center_pose = pose @ np.linalg.inv(to_origin)

            # Draw 3D bounding box
            vis = draw_posed_3d_box(
                reader.K,
                img=vis,
                ob_in_cam=center_pose,
                bbox=bbox
            )

            # Draw XYZ axes
            vis = draw_xyz_axis(
                vis,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True
            )

        # Save visualization for this frame
        out_path = os.path.join(out_dir, f"{frame_id}.png")
        imageio.imwrite(out_path, vis)

    logging.info(f"Done. Visualization images saved to: {out_dir}")


if __name__ == "__main__":
    main()
