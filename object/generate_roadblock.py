import os
import csv
import random
import math
import time
import json
import numpy as np
import carla
from PIL import Image

PORT = 2000

# --- Scene settings ---
CENTER_X, CENTER_Y, CENTER_Z = 92.091, 130.608, 0.2
AREA_HALF = 5.0

# --- Object settings ---
NUM_BLOCKS = 5
RADIUS = 3.0
MAX_SPEED = 1.0
ACCEL_MAX = 0.3

# --- Camera settings ---
RESOLUTION = (1280, 720)
CAM_Z = 2.2
CAM_FOV = 90.0
SAVE_DIR = "output"
SAVE_EVERY_N = 1
TARGET_FPS = 20.0

# --- Depth (metric 16-bit PNG) ---
MAX_DEPTH_M = 1000.0

# --- Pose logging ---
POSE_CSV = "poses.csv"

# --- Static props to use as blocks ---
SMALL_PROP_IDS = [
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
    "static.prop.barrel",
    "static.prop.trashcan02",
    "static.prop.box02",
]
FALLBACK_PATTERNS = ["static.prop.*trafficcone*", "static.prop.*barrel*", "static.prop.*box*"]


def _first_existing_sequence(bp_lib, exact_ids, fallback_patterns, needed):
    chosen, seen = [], set()
    for eid in exact_ids:
        try:
            bp = bp_lib.find(eid)
            if bp and bp.id not in seen:
                chosen.append(bp)
                seen.add(bp.id)
                if len(chosen) >= needed:
                    return chosen
        except RuntimeError:
            pass
    for pat in fallback_patterns:
        for bp in bp_lib.filter(pat):
            if bp.id not in seen:
                chosen.append(bp)
                seen.add(bp.id)
                if len(chosen) >= needed:
                    return chosen
    for bp in bp_lib.filter("static.prop.*"):
        if bp.id not in seen:
            chosen.append(bp)
            seen.add(bp.id)
            if len(chosen) >= needed:
                return chosen
    return chosen


def random_point_on_square_boundary(cx, cy, half):
    side = random.randint(0, 3)
    if side == 0:
        return cx - half, cy + random.uniform(-half, half)
    if side == 1:
        return cx + half, cy + random.uniform(-half, half)
    if side == 2:
        return cx + random.uniform(-half, half), cy - half
    return cx + random.uniform(-half, half), cy + half


def ensure_dirs(W, H):
    base = os.path.join(SAVE_DIR, "cam0", f"{W}x{H}")
    os.makedirs(os.path.join(base, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(base, "depth"), exist_ok=True)
    os.makedirs(os.path.join(base, "mask"), exist_ok=True)
    return base


def compute_intrinsics(width, height, hfov_deg):
    hfov = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return fx, fy, cx, cy, K


def save_intrinsics(out_dir, width, height, hfov_deg):
    fx, fy, cx, cy, K = compute_intrinsics(width, height, hfov_deg)
    with open(os.path.join(out_dir, "intrinsics_cam0.txt"), "w") as f:
        f.write(f"W={width}, H={height}, HFOV_deg={hfov_deg}\n")
        f.write(f"fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}\n")
        f.write("K=\n")
        for row in K:
            f.write("  " + " ".join(f"{v:.6f}" for v in row) + "\n")
    with open(os.path.join(out_dir, "intrinsics_cam0.json"), "w") as f:
        json.dump(
            {"width": width, "height": height, "hfov_deg": hfov_deg,
             "fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K}, f, indent=2
        )
    print(f"📷 Intrinsics saved for {width}x{height}, FOV={hfov_deg}°")


def depth_raw_to_meters(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    d_norm = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0**3 - 1.0)
    return d_norm * 1000.0  # meters


def save_mask_png(path_no_ext, mask_bool):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(path_no_ext + ".png")


def main():
    client = carla.Client('localhost', PORT)
    client.set_timeout(60.0)
    world = client.load_world('Town01')

    # --- Async mode ---
    settings = world.get_settings()
    original = settings
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()

    # === Spawn props ===
    bp_list = _first_existing_sequence(bp_lib, SMALL_PROP_IDS, FALLBACK_PATTERNS, NUM_BLOCKS)
    roadblocks = []
    base_yaw = random.uniform(-180, 180)
    for i in range(NUM_BLOCKS):
        ang = i * (2 * math.pi / NUM_BLOCKS)
        loc = carla.Location(
            x=CENTER_X + RADIUS * math.cos(ang),
            y=CENTER_Y + RADIUS * math.sin(ang),
            z=CENTER_Z
        )
        rot = carla.Rotation(pitch=0.0, yaw=base_yaw + i * 20)
        rb = world.try_spawn_actor(bp_list[i], carla.Transform(loc, rot))
        if rb:
            try: rb.set_simulate_physics(False)
            except: pass
            roadblocks.append(rb)
            print(f"Spawned: {rb.type_id}")
    if not roadblocks:
        print("❌ No blocks spawned.")
        return

    # target mask actor
    target_actor_id = None
    for rb in roadblocks:
        if "trashcan02" in rb.type_id.lower():
            target_actor_id = rb.id
            break
    if not target_actor_id:
        target_actor_id = roadblocks[0].id
        print("⚠️ trashcan02 not found; using first actor for mask.")
    else:
        print(f"🎯 Mask target actor id={target_actor_id}")

    # --- Camera ---
    W, H = RESOLUTION
    base_dir = ensure_dirs(W, H)
    rgb_dir = os.path.join(base_dir, "rgb")
    depth_dir = os.path.join(base_dir, "depth")
    mask_dir = os.path.join(base_dir, "mask")

    cam_x, cam_y = random_point_on_square_boundary(CENTER_X, CENTER_Y, AREA_HALF)
    yaw_to_center = math.degrees(math.atan2(CENTER_Y - cam_y, CENTER_X - cam_x))

    cam_tf = carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=CAM_Z),
        carla.Rotation(pitch=-5.0, yaw=yaw_to_center)
    )

    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(W))
    rgb_bp.set_attribute("image_size_y", str(H))
    rgb_bp.set_attribute("fov", str(CAM_FOV))
    rgb_bp.set_attribute("sensor_tick", str(1.0 / TARGET_FPS))
    rgb_cam = world.spawn_actor(rgb_bp, cam_tf)

    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(W))
    depth_bp.set_attribute("image_size_y", str(H))
    depth_bp.set_attribute("fov", str(CAM_FOV))
    depth_bp.set_attribute("sensor_tick", str(1.0 / TARGET_FPS))
    depth_cam = world.spawn_actor(depth_bp, cam_tf)

    inst_bp = bp_lib.find("sensor.camera.instance_segmentation")
    inst_bp.set_attribute("image_size_x", str(W))
    inst_bp.set_attribute("image_size_y", str(H))
    inst_bp.set_attribute("fov", str(CAM_FOV))
    inst_bp.set_attribute("sensor_tick", str(1.0 / TARGET_FPS))
    inst_cam = world.spawn_actor(inst_bp, cam_tf)

    save_intrinsics(SAVE_DIR, W, H, CAM_FOV)

    # === Callbacks ===
    frame_counter = {"idx": 0}
    saved_mask = {"done": False}

    def rgb_cb(image):
        frame_counter["idx"] += 1
        idx = frame_counter["idx"]
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))
        rgb = arr[:, :, :3][:, :, ::-1]  # BGRA → RGB
        Image.fromarray(rgb, mode="RGB").save(os.path.join(rgb_dir, f"{idx}.png"))

    def depth_cb(image):
        idx = frame_counter["idx"]
        if idx == 0:
            return
        d_m = depth_raw_to_meters(image)
        d_clipped = np.clip(d_m, 0.0, MAX_DEPTH_M)
        d16 = (d_clipped / MAX_DEPTH_M * 65535.0).astype(np.uint16)
        Image.fromarray(d16, mode="I;16").save(os.path.join(depth_dir, f"{idx}.png"))

    def inst_cb(image):
        if saved_mask["done"]:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))
        b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        obj_id = r.astype(np.uint32) + g.astype(np.uint32) * 256 + b.astype(np.uint32) * 256 * 256
        mask = (obj_id == np.uint32(target_actor_id))
        save_mask_png(os.path.join(mask_dir, "mask_trashcan02_frame1"), mask)
        print("✅ Saved trashcan02 mask.")
        saved_mask["done"] = True
        inst_cam.stop()
        inst_cam.destroy()

    rgb_cam.listen(rgb_cb)
    depth_cam.listen(depth_cb)
    inst_cam.listen(inst_cb)

    # --- Object motion setup ---
    positions, velocities = [], []
    for rb in roadblocks:
        loc = rb.get_transform().location
        positions.append([loc.x, loc.y])
        th = random.uniform(0, 2 * math.pi)
        v0 = random.uniform(0.0, 0.5)
        velocities.append([v0 * math.cos(th), v0 * math.sin(th)])

    pose_path = os.path.join(SAVE_DIR, POSE_CSV)
    f_csv = open(pose_path, "w", newline="")
    csvw = csv.writer(f_csv)
    csvw.writerow(["frame", "block_idx", "actor_id", "x", "y", "z", "yaw", "vx", "vy"])

    print("🚧 Running async capture... Stop with Ctrl+C.")
    dt = 1.0 / TARGET_FPS
    last = time.time()

    try:
        while True:
            now = time.time()
            if now - last < dt:
                time.sleep(dt - (now - last))
            last = time.time()

            for i, rb in enumerate(roadblocks):
                x, y = positions[i]
                vx, vy = velocities[i]
                ax = random.uniform(-ACCEL_MAX, ACCEL_MAX)
                ay = random.uniform(-ACCEL_MAX, ACCEL_MAX)
                vx += ax * dt
                vy += ay * dt
                spd = math.hypot(vx, vy)
                if spd > MAX_SPEED:
                    s = MAX_SPEED / spd
                    vx *= s
                    vy *= s
                x += vx * dt
                y += vy * dt
                if x < CENTER_X - AREA_HALF or x > CENTER_X + AREA_HALF:
                    vx = -vx
                if y < CENTER_Y - AREA_HALF or y > CENTER_Y + AREA_HALF:
                    vy = -vy
                positions[i], velocities[i] = [x, y], [vx, vy]
                yaw = math.degrees(math.atan2(vy, vx))
                rb.set_transform(carla.Transform(carla.Location(x, y, CENTER_Z),
                                                 carla.Rotation(yaw=yaw)))
                csvw.writerow([frame_counter["idx"], i, rb.id, x, y, CENTER_Z, yaw, vx, vy])

            if frame_counter["idx"] % 50 == 0:
                f_csv.flush()
            world.wait_for_tick()

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        for s in [rgb_cam, depth_cam]:
            try:
                s.stop()
                s.destroy()
            except:
                pass
        for rb in roadblocks:
            try:
                rb.destroy()
            except:
                pass
        try:
            f_csv.close()
        except:
            pass
        world.apply_settings(original)
        print("✅ Cleaned up. Dataset saved at:", os.path.abspath(SAVE_DIR))


if __name__ == "__main__":
    main()
