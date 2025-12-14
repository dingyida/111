import bpy
import json
import os
import math
import glob
import numpy as np
from mathutils import Euler, Vector

# ========= PATH CONFIG =========
BASE = "/home/yid324"   # <<< change if needed

TRAJ_JSON = os.path.join(BASE, "trajectories.json")
CAM_JSON  = os.path.join(BASE, "cameras.json")

# GLB models (rigid robots)
MODEL_DIR  = BASE
MODEL_FILES = ["1.glb", "2.glb", "3.glb", "4.glb", "5.glb"]

# Render output root (FoundationPose-style sequences will go here)
OUT_ROOT = os.path.join(BASE, "render_output")

# === Recenter CARLA world to Blender origin ===
CENTER_X = 92.091
CENTER_Y = 130.608

# === Motion area radius (for debug circle) ===
AREA_RADIUS = 5.0

# === Frame sampling ===
FRAME_STEP = 10
MAX_RENDER_FRAMES = 200

# === Synthetic camera ring ===
CAM_RING_RADIUS = 12.0
CAM_HEIGHT      = 6.0
CAM_FOV_DEG     = 60.0
NUM_CAMS        = 5

# === Which actors to render ===
ACTIVE_ACTOR_IDS = None  # None -> all actors

# === Per-model manual scale & rotation offset (IMPORTANT) ===
MODEL_SCALE = {
    "1": 1.0,
    "2": 0.3,
    "3": 3.0,
    "4": 1.0,
    "5": 1.0,
}

# Rotation offset in degrees for each GLB model
MODEL_ROT_EULER_DEG = {
    "1": (-90.0, 0.0, 0.0),
    "2": (0.0, 0.0, 0.0),
    "3": (-90.0, 0.0, 0.0),
    "4": (0.0, 0.0, 0.0),
    "5": (0.0, 0.0, 0.0),
}

# ===========================================================
# Utility: look_at + intrinsics
# ===========================================================
def look_at(obj, target: Vector):
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


def compute_K_from_blender(cam_obj, width, height):
    """Compute intrinsics from Blender camera FOV (horizontal)."""
    cam_data = cam_obj.data
    fov = cam_data.angle  # radians, horizontal FOV

    fx = 0.5 * width / math.tan(fov / 2.0)
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


# ---- clear scene ----
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

scene = bpy.context.scene

# ---- render settings ----
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
prefs = bpy.context.preferences
if "cycles" in prefs.addons:
    prefs.addons["cycles"].preferences.compute_device_type = 'CUDA'

scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_depth = '8'
scene.render.resolution_percentage = 100

# Enable passes needed for depth & mask
view_layer = scene.view_layers["ViewLayer"]
view_layer.use_pass_z = True
view_layer.use_pass_object_index = True

# ---- light ----
bpy.ops.object.light_add(type='SUN', radius=1.0,
                         location=(0.0, 0.0, 20.0))
sun = bpy.context.active_object
sun.data.energy = 5.0


# ---- debug: center + circle + ground ----
def add_debug_center_and_circle():
    # Center sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25,
                                         location=(0.0, 0.0, 0.5))
    center_sphere = bpy.context.active_object
    mat_center = bpy.data.materials.new("CenterMat")
    mat_center.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    center_sphere.data.materials.append(mat_center)
    center_sphere.pass_index = 0

    # Motion boundary circle
    bpy.ops.mesh.primitive_circle_add(
        radius=AREA_RADIUS,
        vertices=64,
        fill_type='NOTHING',
        location=(0.0, 0.0, 0.3)
    )
    circle = bpy.context.active_object
    mat_circle = bpy.data.materials.new("AreaCircleMat")
    mat_circle.diffuse_color = (0.0, 1.0, 0.0, 1.0)
    circle.data.materials.append(mat_circle)
    circle.pass_index = 0

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=AREA_RADIUS * 4,
                                     location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    mat_ground = bpy.data.materials.new("GroundMat")
    mat_ground.diffuse_color = (0.2, 0.2, 0.2, 1.0)
    ground.data.materials.append(mat_ground)
    ground.pass_index = 0


add_debug_center_and_circle()


# ---- load JSON ----
print(f"Loading trajectories from {TRAJ_JSON}")
with open(TRAJ_JSON, "r") as f:
    trajectories = json.load(f)

print(f"Loading cameras from {CAM_JSON}")
with open(CAM_JSON, "r") as f:
    cameras_json = json.load(f)

# ---- parse cameras.json (only resolutions per cam_idx) ----
cams_by_idx = {}
for key, c in cameras_json.items():
    idx = int(c["cam_idx"])
    W = int(c["width"])
    H = int(c["height"])
    cams_by_idx.setdefault(idx, {"resolutions": []})
    res = (W, H)
    if res not in cams_by_idx[idx]["resolutions"]:
        cams_by_idx[idx]["resolutions"].append(res)

for idx in cams_by_idx:
    cams_by_idx[idx]["resolutions"].sort()

print("Cameras (from JSON) resolutions:")
for idx, info in cams_by_idx.items():
    print(f"  cam{idx}: resolutions={info['resolutions']}")


# ---- create camera ring ----
cam_objects = {}
cam_indices = sorted(cams_by_idx.keys())
if len(cam_indices) != NUM_CAMS:
    print(f"[WARN] cameras.json has {len(cam_indices)} unique cam_idx, NUM_CAMS={NUM_CAMS}")

for i, idx in enumerate(cam_indices):
    angle = 2.0 * math.pi * i / max(1, len(cam_indices))
    x = CAM_RING_RADIUS * math.cos(angle)
    y = CAM_RING_RADIUS * math.sin(angle)
    z = CAM_HEIGHT

    cam_data = bpy.data.cameras.new(f"Camera_{idx}")
    cam_obj = bpy.data.objects.new(f"Camera_{idx}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_data.angle = math.radians(CAM_FOV_DEG)
    cam_data.clip_start = 0.1
    cam_data.clip_end = 1000.0

    cam_obj.location = (x, y, z)
    look_at(cam_obj, Vector((0.0, 0.0, 0.0)))

    cam_objects[idx] = cam_obj
    print(f"Created cam{idx} at ({x:.2f},{y:.2f},{z:.2f}), looking at origin.")

if cam_objects:
    scene.camera = cam_objects[cam_indices[0]]


# ---- import GLB models -> meshes only ----
base_meshes = {}
model_rot_offset_rad = {}

for fname in MODEL_FILES:
    path = os.path.join(MODEL_DIR, fname)
    print(f"\nImporting GLB: {path}")
    if not os.path.isfile(path):
        print(f"[WARN] GLB not found: {path}")
        continue

    bpy.ops.import_scene.gltf(filepath=path)
    imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported:
        print(f"[WARN] No mesh objects found in {fname}")
        continue

    if len(imported) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
        base_obj = imported[0]
    else:
        base_obj = imported[0]

    model_name = os.path.splitext(fname)[0]  # "1","2","3","4","5"
    scale_factor = MODEL_SCALE.get(model_name, 1.0)
    base_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.objects.active = base_obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mesh = base_obj.data
    mesh.name = f"Mesh_{model_name}"

    bpy.data.objects.remove(base_obj, do_unlink=True)

    base_meshes[model_name] = mesh

    degs = MODEL_ROT_EULER_DEG.get(model_name, (0.0, 0.0, 0.0))
    model_rot_offset_rad[model_name] = tuple(math.radians(d) for d in degs)

    print(f"Loaded mesh for model '{model_name}' from {fname}, scale={scale_factor}, rot_deg={degs}")

if not base_meshes:
    raise RuntimeError("No base meshes loaded. Check MODEL_FILES paths.")

print("\nLoaded GLB meshes:", list(base_meshes.keys()))


# ---- actor ¡ú model mapping ----
all_actor_ids = sorted(trajectories.keys(), key=lambda x: int(x))

if ACTIVE_ACTOR_IDS is None:
    actor_ids = all_actor_ids
else:
    wanted = set(str(x) for x in ACTIVE_ACTOR_IDS)
    actor_ids = [aid for aid in all_actor_ids if aid in wanted]

if len(actor_ids) == 0:
    raise RuntimeError("No actors selected for rendering!")

print("Actors present in trajectories:", all_actor_ids)
print("Actors selected for rendering:", actor_ids)

model_names = list(base_meshes.keys())  # ["1","2","3","4","5"]
actor_to_model = {aid: model_names[i % len(model_names)]
                  for i, aid in enumerate(actor_ids)}

print("\n================ ACTOR ¡ú MODEL MAPPING ================")
for aid in actor_ids:
    print(f"  Actor {aid}  -->  {actor_to_model[aid]}.glb")
print("=======================================================\n")


# ---- assign a unique mask index for each model (5 objects -> 5 masks) ----
model_to_pass_index = {}
for idx, model_name in enumerate(sorted(base_meshes.keys()), start=1):
    model_to_pass_index[model_name] = idx  # 1..5


# ---- instantiate actors ----
actor_objects = {}
actor_rot_offset = {}

for actor_id, model_name in actor_to_model.items():
    mesh = base_meshes[model_name]
    obj = bpy.data.objects.new(f"Actor_{actor_id}", mesh)
    bpy.context.scene.collection.objects.link(obj)

    obj.pass_index = model_to_pass_index[model_name]  # mask index per model

    actor_objects[actor_id] = obj
    actor_rot_offset[actor_id] = model_rot_offset_rad[model_name]


# ---- build frame index using the first actor ----
first_actor_id = actor_ids[0]
frames_list = trajectories[first_actor_id]["frames"]
num_samples = len(frames_list)
frame_ids = [f["frame"] for f in frames_list]

print(f"\nTotal trajectory samples: {num_samples}")

indices = list(range(0, num_samples, FRAME_STEP))
if MAX_RENDER_FRAMES and MAX_RENDER_FRAMES > 0:
    indices = indices[:MAX_RENDER_FRAMES]

print(f"Will render {len(indices)} frames (step={FRAME_STEP}, max={MAX_RENDER_FRAMES})")

scene.frame_start = 1
scene.frame_end = len(indices)


# ===========================================================
# Compositor setup: Depth EXR + 5 object masks
# ===========================================================
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

rlayers = tree.nodes.new("CompositorNodeRLayers")

# Depth output: EXR, 32-bit float
depth_out = tree.nodes.new("CompositorNodeOutputFile")
depth_out.name = "DepthRaw"
depth_out.format.file_format = "OPEN_EXR"
depth_out.format.color_depth = "32"
depth_out.file_slots[0].use_node_format = True
tree.links.new(rlayers.outputs["Depth"], depth_out.inputs["Image"])

# ID-mask outputs for 5 objects (models)
mask_nodes = {}
for idx in range(1, 6):   # pass_index 1..5
    idmask = tree.nodes.new("CompositorNodeIDMask")
    idmask.index = idx
    tree.links.new(rlayers.outputs["IndexOB"], idmask.inputs["ID value"])

    mask_out = tree.nodes.new("CompositorNodeOutputFile")
    mask_out.name = f"MaskObj{idx}"
    mask_out.format.file_format = "PNG"
    mask_out.format.color_mode = "BW"
    mask_out.format.color_depth = "8"
    mask_out.file_slots[0].use_node_format = True
    tree.links.new(idmask.outputs["Alpha"], mask_out.inputs["Image"])

    mask_nodes[idx] = mask_out


# ===========================================================
# Prepare FoundationPose-style dirs + cam_K.txt
# ===========================================================
fp_seq_info = {}  # (cam_idx, W, H) -> dict

for cam_idx, info in cams_by_idx.items():
    for (W, H) in info["resolutions"]:
        seq_root = os.path.join(OUT_ROOT, f"fp_cam{cam_idx}_{W}x{H}")
        rgb_dir   = os.path.join(seq_root, "rgb")
        depth_dir = os.path.join(seq_root, "depth")
        masks_root = os.path.join(seq_root, "masks")
        os.makedirs(rgb_dir,   exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(masks_root, exist_ok=True)

        # Write cam_K.txt (3x3)
        cam_obj = cam_objects[cam_idx]
        K = compute_K_from_blender(cam_obj, W, H)
        camK_path = os.path.join(seq_root, "cam_K.txt")
        if not os.path.exists(camK_path):
            np.savetxt(camK_path, K.reshape(1, 9))
            print(f"[cam{cam_idx} {W}x{H}] wrote {camK_path}:\n{K}")

        fp_seq_info[(cam_idx, W, H)] = {
            "root": seq_root,
            "rgb": rgb_dir,
            "depth": depth_dir,
            "masks_root": masks_root,
        }


# ===========================================================
# Main render loop
# ===========================================================
for count, i in enumerate(indices):
    frame_id = frame_ids[i]
    frame_idx = count + 1
    scene.frame_set(frame_idx)

    # 6-digit id string: 000001, 000002, ...
    id_str = f"{frame_idx:06d}"

    # Update actor positions & rotations
    for actor_id in actor_ids:
        info = trajectories[actor_id]
        f = info["frames"][i]
        x_world = f["x"] - CENTER_X
        y_world = f["y"] - CENTER_Y
        z_world = f["z"]
        yaw_deg = f["yaw"]

        obj = actor_objects[actor_id]
        rx, ry, rz = actor_rot_offset[actor_id]
        obj.location = (x_world, y_world, z_world)
        obj.rotation_euler = Euler((rx, ry, rz + math.radians(yaw_deg)))

    # Render for each camera & resolution
    for cam_idx, cam_obj in cam_objects.items():
        scene.camera = cam_obj
        cam_info = cams_by_idx[cam_idx]

        for (W, H) in cam_info["resolutions"]:
            scene.render.resolution_x = W
            scene.render.resolution_y = H

            seq = fp_seq_info[(cam_idx, W, H)]
            rgb_dir   = seq["rgb"]
            depth_dir = seq["depth"]
            masks_root = seq["masks_root"]

            # ----- RGB: explicit filename -----
            scene.render.filepath = os.path.join(rgb_dir, f"{id_str}.png")

            # ----- Depth EXR -----
            depth_out.base_path = depth_dir
            depth_out.file_slots[0].path = id_str  # depth/000001.exr

            # ----- Masks (obj1..obj5) -----
            for pidx in range(1, 6):
                obj_mask_dir = os.path.join(masks_root, f"obj{pidx}")
                os.makedirs(obj_mask_dir, exist_ok=True)
                mask_nodes[pidx].base_path = obj_mask_dir
                mask_nodes[pidx].file_slots[0].path = id_str  # masks/objX/000001.png

            bpy.ops.render.render(write_still=True)
            print(f"[{count+1}/{len(indices)}] frame {frame_id} cam{cam_idx} {W}x{H}")

print("Rendering finished.")
print("RGB (PNG), depth (EXR) and masks (PNG) are under:", OUT_ROOT)
print("Next step: convert depth/*.exr -> uint16 mm PNG in your normal Python env.")

