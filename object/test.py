# bounded_scene_2cams.py
# CARLA 0.9.15 | Python 3.7 (64-bit, Windows)
import os, math, random, time
import carla

# ------------ Tunables (safe defaults) ------------
NUM_VEHICLES  = 5
SIM_SECONDS   = 5
FIXED_DT      = 0.05        # 20 FPS simulation
IMG_W, IMG_H  = 360, 180    # reduce if VRAM is tight (e.g., 320x180)
SENSOR_FPS    = 10.0        # camera FPS (lower = lighter)
USE_CAMS      = 2           # <= requested: only 2 cameras
MAX_SPEED     = 10.0        # m/s cap in our simple controller
TARGET_RADIUS = 5.0
STEER_P       = 0.02
CLIENT_TIMEOUT_S = 60.0
TICK_TIMEOUT_BACKOFF_S = 0.2
# --------------------------------------------------

def ensure_dirs(ncams):
    os.makedirs("output", exist_ok=True)
    for i in range(ncams):
        os.makedirs(os.path.join("output", f"camera_{i}"), exist_ok=True)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def pick_random_point(xmin, xmax, ymin, ymax, z):
    return carla.Location(x=random.uniform(xmin, xmax),
                          y=random.uniform(ymin, ymax),
                          z=z)

def drive_to_target(v, tgt, max_speed, steer_p):
    tf = v.get_transform()
    loc, yaw = tf.location, tf.rotation.yaw
    dx, dy = tgt.x - loc.x, tgt.y - loc.y
    dist = (dx*dx + dy*dy) ** 0.5
    desired = math.degrees(math.atan2(dy, dx))
    err = (desired - yaw + 180.0) % 360.0 - 180.0
    steer = clamp(steer_p * err, -1.0, 1.0)

    vel = v.get_velocity()
    speed = (vel.x * vel.x + vel.y * vel.y) ** 0.5
    throttle = 0.0 if speed > max_speed else clamp(0.3 + 0.4 * min(dist / 30.0, 1.0), 0.3, 0.7)
    v.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))
    return dist

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(CLIENT_TIMEOUT_S)
    world  = client.get_world()

    # Define a 120m x 120m box around the first spawn point
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points found on the current map.")
    center = spawns[0].location
    cx, cy, cz = center.x, center.y, center.z
    print(f"Center location: x={cx:.2f}, y={cy:.2f}, z={cz:.2f}")
    half_w, half_h = 60.0, 60.0
    x_min, x_max = cx - half_w, cx + half_w
    y_min, y_max = cy - half_h, cy + half_h

    orig = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    ensure_dirs(USE_CAMS)
    bp_lib = world.get_blueprint_library()
    veh_bps = bp_lib.filter("vehicle.*")

    vehicles, targets, cameras = [], {}, []
    try:
        # Spawn vehicles inside the box
        random.shuffle(spawns)
        for sp in spawns:
            if len(vehicles) >= NUM_VEHICLES:
                break
            loc = sp.location
            if x_min <= loc.x <= x_max and y_min <= loc.y <= y_max:
                bp = random.choice(veh_bps)
                v = world.try_spawn_actor(bp, sp)
                if v:
                    v.set_autopilot(False)
                    vehicles.append(v)
                    targets[v.id] = pick_random_point(x_min, x_max, y_min, y_max, loc.z)

        if not vehicles:
            raise RuntimeError("Failed to spawn vehicles in the box. Enlarge the box or change map area.")
        print(f"Spawned {len(vehicles)} vehicles near ({cx:.1f},{cy:.1f}).")

        # Lightweight camera blueprint
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", f"{1.0/max(SENSOR_FPS,1.0):.3f}")  # seconds per frame
        cam_bp.set_attribute("enable_postprocess_effects", "False")

        # Two cameras: North (looking south) and top-down
        cam_tfs = [
            carla.Transform(carla.Location(x=cx, y=y_max + 20.0, z=25.0), carla.Rotation(pitch=-15.0, yaw=180.0)),
            carla.Transform(carla.Location(x=cx, y=cy,              z=70.0), carla.Rotation(pitch=-90.0, yaw=0.0)),
        ]

        for idx, tf in enumerate(cam_tfs[:USE_CAMS]):
            cam = world.spawn_actor(cam_bp, tf)
            def _cb(image, i=idx):
                image.save_to_disk(os.path.join("output", f"camera_{i}", f"frame_{image.frame:06d}.png"))
            cam.listen(_cb)
            cameras.append(cam)

        print(f"Cameras spawned: {len(cameras)} @ {IMG_W}x{IMG_H} ~{int(SENSOR_FPS)} FPS.")

        ticks = int(SIM_SECONDS / FIXED_DT)
        for step in range(ticks):
            try:
                world.tick()
            except RuntimeError as e:
                # CARLA 0.9.15 raises RuntimeError for tick timeouts
                if "time-out" in str(e).lower():
                    print("[WARN] world.tick() timed out; backing off and continuing…")
                    time.sleep(TICK_TIMEOUT_BACKOFF_S)
                    continue
                else:
                    raise

            # Keep vehicles inside the box, wandering toward random targets
            for v in vehicles:
                loc = v.get_transform().location
                if not (x_min <= loc.x <= x_max and y_min <= loc.y <= y_max):
                    targets[v.id] = pick_random_point(x_min, x_max, y_min, y_max, loc.z)

                dist = drive_to_target(v, targets[v.id], MAX_SPEED, STEER_P)
                if dist < TARGET_RADIUS:
                    targets[v.id] = pick_random_point(x_min, x_max, y_min, y_max, loc.z)

            time.sleep(0.001)

        print("Simulation finished.")

    finally:
        print("Cleaning up…")
        # Stop and destroy cameras
        for cam in cameras:
            try:
                cam.stop()
            except Exception:
                pass
            try:
                cam.destroy()
            except Exception:
                pass
        # Destroy vehicles
        for v in vehicles:
            try:
                v.destroy()
            except Exception:
                pass
        # Restore settings
        world.apply_settings(orig)
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
