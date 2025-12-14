import os
import csv
import random
import math
import carla

PORT = 2000

# === Output mode switch ===
EXPORT_ONLY_CSV = True


# --- Motion center & movement radius (CIRCLE) ---
CENTER_X, CENTER_Y = 92.091, 130.608
CENTER_Z = 0.2
AREA_RADIUS = 5.0  # objects can only move within this radius (meters)

# --- Blocks ---
NUM_BLOCKS = 5
RADIUS = 3.0          # initial spawn radius around center
MAX_SPEED = 3.0       # max linear speed (m/s)

# === PER-OBJECT ACCELERATION CONFIG ===
# Block index -> max random acceleration (m/s^2)
BLOCK_ACCEL_CFG = {
    0: 0.3,
    1: 1.0,
    2: 2.5,
    3: 0.5,
    4: 1.8,
}

# --- Collision avoidance ---
MIN_DIST = 1.0      # minimum separation distance (m)
SEP_GAIN = 2.0      # strength of separation force

# --- Cameras ---
NUM_CAMERAS = 5
RESOLUTIONS = [(320, 180), (640, 360), (1280, 720)]  # 3 outputs per camera
CAM_Z = 2.2
CAM_FOV = 90.0
SAVE_EVERY_N = 1
IMAGE_EXT = "jpg"

# --- CSV output ---
SAVE_DIR = "output"
POSE_CSV = "poses.csv"
CAMERA_CSV = "cameras.csv"

# Block mesh blueprints
SMALL_PROP_IDS = [
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
    "static.prop.barrel",
    "static.prop.trashcan02",
    "static.prop.box02",
]

FALLBACK_PATTERNS = [
    "static.prop.*trafficcone*",
    "static.prop.*barrel*",
    "static.prop.*box*",
]


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


def random_point_on_circle_boundary(cx, cy, radius):
    ang = random.uniform(0.0, 2.0 * math.pi)
    return cx + radius * math.cos(ang), cy + radius * math.sin(ang)


def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 为相机图像建目录（即使 CSV-only 模式也没关系）
    for i in range(NUM_CAMERAS):
        for (w, h) in RESOLUTIONS:
            os.makedirs(os.path.join(SAVE_DIR, f"cam{i}", f"{w}x{h}"), exist_ok=True)


def main():
    client = carla.Client('localhost', PORT)
    client.set_timeout(60.0)
    world = client.load_world('Town01')

    bp_lib = world.get_blueprint_library()
    bp_list = _first_existing_sequence(
        bp_lib, SMALL_PROP_IDS, FALLBACK_PATTERNS, NUM_BLOCKS
    )

    roadblocks = []
    positions = []   # [x, y] in world coords
    velocities = []  # [vx, vy]
    actor_ids = []

    print("\n=== Block Acceleration Profile ===")
    for i in range(NUM_BLOCKS):
        accel = BLOCK_ACCEL_CFG.get(i, 1.0)
        print(f"Block {i}: ACCEL_MAX = {accel}")
    print()

    # === Spawn objects around center on a circle of radius RADIUS ===
    base_yaw = random.uniform(-180, 180)
    for i in range(NUM_BLOCKS):
        ang = i * (2 * math.pi / NUM_BLOCKS)
        dx = RADIUS * math.cos(ang)
        dy = RADIUS * math.sin(ang)
        loc = carla.Location(
            x=CENTER_X + dx,
            y=CENTER_Y + dy,
            z=CENTER_Z
        )
        rot = carla.Rotation(pitch=0.0, yaw=base_yaw + i * 20)

        rb = world.try_spawn_actor(bp_list[i], carla.Transform(loc, rot))
        if rb:
            try:
                rb.set_simulate_physics(False)
            except:
                pass
            if hasattr(rb, "enable_gravity"):
                try:
                    rb.enable_gravity(False)
                except:
                    pass

            roadblocks.append(rb)
            positions.append([loc.x, loc.y])

            # small random initial velocity
            th0 = random.uniform(0, 2 * math.pi)
            v0 = random.uniform(0.0, 0.5)
            velocities.append([v0 * math.cos(th0), v0 * math.sin(th0)])
            actor_ids.append(rb.id)

            print(
                f"Spawned block {i}: {rb.type_id} "
                f"@ ({loc.x:.2f}, {loc.y:.2f})"
            )
        else:
            print(f"⚠️ Failed to spawn block {i}")

    if not roadblocks:
        print("❌ No roadblocks spawned.")
        return

    ensure_dirs()

    # === Spawn cameras on circle boundary, looking at center ===
    cameras = []  # list of (cam_actor, cam_idx, (W,H))
    cam_positions = [
        random_point_on_circle_boundary(CENTER_X, CENTER_Y, AREA_RADIUS * 1.2)
        for _ in range(NUM_CAMERAS)
    ]

    for i, (x, y) in enumerate(cam_positions):
        # yaw that looks toward the center
        yaw_deg = math.degrees(math.atan2(CENTER_Y - y, CENTER_X - x))

        for (W, H) in RESOLUTIONS:
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(W))
            cam_bp.set_attribute("image_size_y", str(H))
            cam_bp.set_attribute("fov", str(CAM_FOV))
            cam_bp.set_attribute("sensor_tick", str(1.0 / 20.0))  # ~20 FPS

            cam_tf = carla.Transform(
                carla.Location(x=x, y=y, z=CAM_Z),
                carla.Rotation(pitch=-5.0, yaw=yaw_deg)
            )
            cam = world.spawn_actor(cam_bp, cam_tf)
            cameras.append((cam, i, (W, H)))

            print(
                f"Spawned cam{i} [{W}x{H}] "
                f"@ ({x:.2f},{y:.2f},{CAM_Z:.2f}) yaw→center={yaw_deg:.1f}°"
            )

    # === 写入 camera CSV（每个物理相机每个分辨率一行） ===
    cam_csv_path = os.path.join(SAVE_DIR, CAMERA_CSV)
    with open(cam_csv_path, 'w', newline='') as f_cam:
        camw = csv.writer(f_cam)
        camw.writerow(
            ["cam_idx", "width", "height", "fov",
             "x", "y", "z", "yaw", "pitch", "roll"]
        )
        for cam, idx, (W, H) in cameras:
            tf = cam.get_transform()
            loc, rot = tf.location, tf.rotation
            camw.writerow([
                idx, W, H, CAM_FOV,
                loc.x, loc.y, loc.z,
                rot.yaw, rot.pitch, rot.roll])

    print(f"📷 Camera parameters saved to: {os.path.abspath(cam_csv_path)}")

    # === 相机回调（仅在导出图像模式下开启监听） ===
    def make_cam_callback(cam_idx, size_tuple):
        W, H = size_tuple
        subdir = os.path.join(SAVE_DIR, f"cam{cam_idx}", f"{W}x{H}")

        def _cb(image):
            # 纯 CSV 模式：不保存图片
            if EXPORT_ONLY_CSV:
                return
            if image.frame % SAVE_EVERY_N != 0:
                return
            path = os.path.join(subdir, f"frame_{image.frame:08d}.{IMAGE_EXT}")
            image.save_to_disk(path, carla.ColorConverter.Raw)
        return _cb

    if not EXPORT_ONLY_CSV:
        for cam, idx, sz in cameras:
            cam.listen(make_cam_callback(idx, sz))
        print("📸 Camera listening enabled (full export mode).")
    else:
        print("📝 EXPORT_ONLY_CSV = True，仅导出 CSV，不保存图像。")

    # === Pose CSV ===
    pose_path = os.path.join(SAVE_DIR, POSE_CSV)
    f_csv = open(pose_path, 'w', newline='')
    csvw = csv.writer(f_csv)
    csvw.writerow([
        "frame", "block_idx", "actor_id",
        "x", "y", "z",
        "yaw",
        "vx", "vy", "speed",
        "accel_limit"
    ])

    print("\nSimulation running within a circle (radius = "
          f"{AREA_RADIUS} m). Ctrl+C to stop.\n")

    try:
        while True:
            snapshot = world.wait_for_tick()
            dt = snapshot.timestamp.delta_seconds
            frame = snapshot.frame
            if dt <= 0:
                dt = 1.0 / 20.0

            prev_positions = [p[:] for p in positions]

            for i, rb in enumerate(roadblocks):
                x, y = prev_positions[i]
                vx, vy = velocities[i]

                # === Object-specific max acceleration ===
                accel_max = BLOCK_ACCEL_CFG.get(i, 1.0)

                # Random accel direction with object-specific magnitude
                a_mag = random.uniform(0.0, accel_max)
                a_th = random.uniform(0.0, 2.0 * math.pi)
                ax = a_mag * math.cos(a_th)
                ay = a_mag * math.sin(a_th)

                # === Collision avoidance (repulsive acceleration) ===
                for j in range(len(roadblocks)):
                    if j == i:
                        continue
                    ox, oy = prev_positions[j]
                    dx = x - ox
                    dy = y - oy
                    dist = math.hypot(dx, dy)
                    if dist < MIN_DIST and dist > 1e-3:
                        strength = SEP_GAIN * (MIN_DIST - dist) / MIN_DIST
                        ax += strength * (dx / dist)
                        ay += strength * (dy / dist)

                # Integrate velocity
                vx += ax * dt
                vy += ay * dt

                speed = math.hypot(vx, vy)
                if speed > MAX_SPEED:
                    s = MAX_SPEED / speed
                    vx *= s
                    vy *= s
                    speed = MAX_SPEED

                # Integrate position
                x += vx * dt
                y += vy * dt

                # === Enforce circular boundary around (CENTER_X, CENTER_Y) ===
                dx_c = x - CENTER_X
                dy_c = y - CENTER_Y
                r = math.hypot(dx_c, dy_c)

                if r > AREA_RADIUS and r > 1e-6:
                    # Project back to the circle boundary
                    scale = AREA_RADIUS / r
                    x = CENTER_X + dx_c * scale
                    y = CENTER_Y + dy_c * scale

                    # Reflect velocity across the radial normal
                    n_x = dx_c / r
                    n_y = dy_c / r
                    v_rad = vx * n_x + vy * n_y
                    vx = vx - 2.0 * v_rad * n_x
                    vy = vy - 2.0 * v_rad * n_y

                    speed = math.hypot(vx, vy)
                    if speed > MAX_SPEED:
                        s2 = MAX_SPEED / speed
                        vx *= s2
                        vy *= s2
                        speed = MAX_SPEED

                # Update state arrays
                positions[i] = [x, y]
                velocities[i] = [vx, vy]

                # Orientation from velocity
                yaw = math.degrees(math.atan2(vy, vx)) if speed > 1e-3 else 0.0

                # Apply transform in CARLA
                rb.set_transform(
                    carla.Transform(
                        carla.Location(x, y, CENTER_Z),
                        carla.Rotation(yaw=yaw)
                    )
                )

                # Log pose to CSV
                csvw.writerow([
                    frame, i, actor_ids[i],
                    x, y, CENTER_Z,
                    yaw,
                    vx, vy, speed,
                    accel_max
                ])

            # flush occasionally
            if frame % 50 == 0:
                f_csv.flush()

    except KeyboardInterrupt:
        print("\nStopping simulation...")

    finally:
        # stop and destroy cameras
        for cam, _, _ in cameras:
            try:
                cam.stop()
            except:
                pass
        for cam, _, _ in cameras:
            try:
                cam.destroy()
            except:
                pass

        # destroy blocks
        for rb in roadblocks:
            try:
                rb.destroy()
            except:
                pass

        try:
            f_csv.flush()
            f_csv.close()
        except Exception:
            pass

        print("✅ Cleaned up.")
        print(f"Poses CSV:   {os.path.abspath(pose_path)}")
        print(f"Camera CSV:  {os.path.abspath(cam_csv_path)}")


if __name__ == "__main__":
    main()
