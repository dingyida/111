import os
import csv
import random
import math
import carla
import time

PORT = 2000

# --- Motion area (10x10 square) ---
CENTER_X, CENTER_Y = 92.091, 130.608
CENTER_Z = 0.2
AREA_HALF = 5.0

# --- Blocks ---
NUM_BLOCKS = 5
RADIUS = 3.0  # initial spawn radius
MAX_SPEED = 3.0     # m/s
ACCEL_MAX = 1.0     # m/s^2 per frame (random 0..1)

# --- Cameras ---
NUM_CAMERAS = 5
RESOLUTIONS = [(320,180), (640,360), (1280,720)]  # 3 outputs per camera
CAM_Z = 2.2
CAM_FOV = 90.0
SAVE_DIR = "output"
SAVE_EVERY_N = 1                 # save camera images every N frames
IMAGE_EXT = "jpg"                # or "png"

# --- Pose logging ---
POSE_CSV = "poses.csv"
LOG_EVERY_N = 1                  # log every N frames (1 = every frame)

# Small props to use as blocks (your provided list)
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
                chosen.append(bp); seen.add(bp.id)
                if len(chosen) >= needed: return chosen
        except RuntimeError:
            pass
    for pat in fallback_patterns:
        for bp in bp_lib.filter(pat):
            if bp.id not in seen:
                chosen.append(bp); seen.add(bp.id)
                if len(chosen) >= needed: return chosen
    for bp in bp_lib.filter("static.prop.*"):
        if bp.id not in seen:
            chosen.append(bp); seen.add(bp.id)
            if len(chosen) >= needed: return chosen
    if not chosen:
        raise RuntimeError("No static.prop.* blueprints found.")
    return chosen

def random_point_on_square_boundary(cx, cy, half):
    side = random.randint(0, 3)
    if side == 0:   # left
        return cx - half, cy + random.uniform(-half, half)
    if side == 1:   # right
        return cx + half, cy + random.uniform(-half, half)
    if side == 2:   # bottom
        return cx + random.uniform(-half, half), cy - half
    return cx + random.uniform(-half, half), cy + half  # top

def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i in range(NUM_CAMERAS):
        for (w,h) in RESOLUTIONS:
            os.makedirs(os.path.join(SAVE_DIR, f"cam{i}", f"{w}x{h}"), exist_ok=True)

def main():
    client = carla.Client('localhost', PORT)
    client.set_timeout(60.0)  # tolerant in async + heavy I/O
    world = client.load_world('Town01')

    # --- ASYNC MODE ---
    settings = world.get_settings()
    original = settings
    settings.synchronous_mode = False   # async
    settings.fixed_delta_seconds = None # let engine decide
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()

    # === Spawn blocks near center ===
    bp_list = _first_existing_sequence(bp_lib, SMALL_PROP_IDS, FALLBACK_PATTERNS, NUM_BLOCKS)
    roadblocks = []
    base_yaw = random.uniform(-180, 180)
    for i in range(NUM_BLOCKS):
        ang = i * (2 * math.pi / NUM_BLOCKS)
        dx, dy = RADIUS * math.cos(ang), RADIUS * math.sin(ang)
        loc = carla.Location(x=CENTER_X + dx, y=CENTER_Y + dy, z=CENTER_Z)
        rot = carla.Rotation(pitch=0.0, yaw=base_yaw + i * 20)
        rb = world.try_spawn_actor(bp_list[i], carla.Transform(loc, rot))
        if rb:
            try: rb.set_simulate_physics(False)
            except: pass
            if hasattr(rb, "enable_gravity"):
                try: rb.enable_gravity(False)
                except: pass
            roadblocks.append(rb)
            print(f"Spawned block {i+1}: {rb.type_id} @ ({loc.x:.2f}, {loc.y:.2f})")
        else:
            print(f"⚠️ Failed to spawn block {i+1}")

    if not roadblocks:
        print("❌ No roadblocks spawned.")
        return

    # === Spectator overview ===
    spectator = world.get_spectator()
    avg_loc = sum((rb.get_transform().location for rb in roadblocks), carla.Location()) * (1.0 / len(roadblocks))
    spectator.set_transform(carla.Transform(avg_loc + carla.Location(x=-10,y=-10,z=7),
                                            carla.Rotation(pitch=-25,yaw=45)))

    # === Spawn cameras on boundary (3 resolutions per location) ===
    ensure_dirs()
    cameras = []  # list of (cam_actor, cam_idx, (W,H))
    cam_positions = [random_point_on_square_boundary(CENTER_X, CENTER_Y, AREA_HALF)
                     for _ in range(NUM_CAMERAS)]
    for i, (x, y) in enumerate(cam_positions):
        yaw_deg = math.degrees(math.atan2(CENTER_Y - y, CENTER_X - x))
        for (W, H) in RESOLUTIONS:
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(W))
            cam_bp.set_attribute("image_size_y", str(H))
            cam_bp.set_attribute("fov", str(CAM_FOV))
            cam_bp.set_attribute("sensor_tick", str(1.0/20.0))  # ~20 FPS capture
            cam_tf = carla.Transform(carla.Location(x=x, y=y, z=CAM_Z),
                                     carla.Rotation(pitch=-5.0, yaw=yaw_deg))
            cam = world.spawn_actor(cam_bp, cam_tf)
            cameras.append((cam, i, (W, H)))
            print(f"Spawned cam{i} [{W}x{H}] at ({x:.2f},{y:.2f},{CAM_Z:.2f}) yaw→center={yaw_deg:.1f}°")

    # === Camera callbacks ===
    def make_cam_callback(cam_idx, size_tuple):
        W, H = size_tuple
        subdir = os.path.join(SAVE_DIR, f"cam{cam_idx}", f"{W}x{H}")
        def _cb(image):
            if image.frame % SAVE_EVERY_N != 0:
                return
            path = os.path.join(subdir, f"frame_{image.frame:08d}.{IMAGE_EXT}")
            image.save_to_disk(path, carla.ColorConverter.Raw)
        return _cb

    for cam, idx, sz in cameras:
        cam.listen(make_cam_callback(idx, sz))

    # --- Per-block kinematics (async) ---
    positions, velocities, actor_ids = [], [], []
    for rb in roadblocks:
        loc = rb.get_transform().location
        positions.append([loc.x, loc.y])
        th = random.uniform(0, 2*math.pi)
        v0 = random.uniform(0.0, 0.5)
        velocities.append([v0*math.cos(th), v0*math.sin(th)])
        actor_ids.append(rb.id)

    # --- Pose logger setup ---
    pose_path = os.path.join(SAVE_DIR, POSE_CSV)
    f_csv = open(pose_path, 'w', newline='')
    csvw = csv.writer(f_csv)
    csvw.writerow(["frame","block_idx","actor_id","x","y","z","yaw","pitch","roll","vx","vy","speed"])

    print("🚧 Async: random-accel blocks in 10×10 area + 5×(3 res) cams; logging poses each frame.")
    target_fps = 20.0
    dt = 1.0 / target_fps
    last = time.time()
    frame_idx = 0

    try:
        while True:
            # wall-clock pacing for ~20 Hz updates
            now = time.time()
            elapsed = now - last
            if elapsed < dt:
                time.sleep(dt - elapsed)
            last = time.time()
            frame_idx += 1

            # Update blocks using dt
            for i, rb in enumerate(roadblocks):
                x, y   = positions[i]
                vx, vy = velocities[i]

                # random accel (0..1 m/s^2) each iteration
                a_mag = random.uniform(0.0, ACCEL_MAX)
                a_th  = random.uniform(0.0, 2*math.pi)
                ax = a_mag * math.cos(a_th)
                ay = a_mag * math.sin(a_th)

                vx += ax * dt
                vy += ay * dt

                spd = math.hypot(vx, vy)
                if spd > MAX_SPEED:
                    s = MAX_SPEED / spd
                    vx *= s; vy *= s
                    spd = MAX_SPEED

                x += vx * dt
                y += vy * dt

                # reflect on square boundary
                min_x, max_x = CENTER_X - AREA_HALF, CENTER_X + AREA_HALF
                min_y, max_y = CENTER_Y - AREA_HALF, CENTER_Y + AREA_HALF
                if x < min_x: x = min_x; vx = -vx
                elif x > max_x: x = max_x; vx = -vx
                if y < min_y: y = min_y; vy = -vy
                elif y > max_y: y = max_y; vy = -vy

                positions[i]  = [x, y]
                velocities[i] = [vx, vy]

                yaw_deg  = (math.degrees(math.atan2(vy, vx)) if spd > 1e-3 else 0.0)
                pitch_deg, roll_deg = 0.0, 0.0

                # apply transform
                rb.set_transform(carla.Transform(carla.Location(x, y, CENTER_Z),
                                                 carla.Rotation(yaw=yaw_deg)))

                # LOG pose (every N)
                if frame_idx % LOG_EVERY_N == 0:
                    csvw.writerow([
                        frame_idx, i, actor_ids[i],
                        x, y, CENTER_Z,
                        yaw_deg, pitch_deg, roll_deg,
                        vx, vy, spd
                    ])

            # flush CSV occasionally
            if frame_idx % 50 == 0:
                f_csv.flush()

            # in async, just wait for next server tick (non-blocking for us)
            world.wait_for_tick()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # stop/destroy sensors first
        for cam, _, _ in cameras:
            try: cam.stop()
            except: pass
        for cam, _, _ in cameras:
            try: cam.destroy()
            except: pass
        for rb in roadblocks:
            try: rb.destroy()
            except: pass
        try:
            world.apply_settings(original)
        except Exception as e:
            print(f"[WARN] restore settings failed: {e}")
        try:
            f_csv.flush(); f_csv.close()
        except Exception:
            pass
        print(f"✅ Cleaned up.\nImages: {os.path.abspath(SAVE_DIR)}\nPoses CSV: {os.path.abspath(pose_path)}")

if __name__ == "__main__":
    main()
