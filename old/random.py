import carla
import random
import time
import json
import math
import queue
import numpy as np
from pathlib import Path

# ================== CONFIG ==================
TOTAL_FRAMES          = 20
FPS                   = 60
OUTPUT_DIR            = "output"
NUM_WALKERS           = 6
CAM_HEIGHT            = 15.0
CAM_FOV_DEG           = 90.0
IMAGE_WIDTH           = 800
IMAGE_HEIGHT          = 600
SPAWN_RADIUS          = 12.0       # Walkers kept (initially) inside this
RELOCATE_RADIUS       = 13.0       # If beyond, give new destination
ENABLE_WALKER_CAMERAS = True
WALKER_MAX_SPEED      = 1.2
BOUNDARY_CHECK_EVERY  = 10
WRITE_PROJECTION_DATA = True
DRAW_BOUNDARY         = True
USE_BATCH_SPAWN       = False      # Set True to use apply_batch_sync approach
TOWN_NAME             = None       # e.g. "Town03" to force load; None = keep current

# ================== HELPERS ==================

def build_camera_intrinsics(width, height, fov_deg):
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

def world_to_camera_matrix(transform):
    M = transform.get_inverse_matrix()
    return np.array(M, dtype=np.float32).reshape(4, 4)

def project_world_point(p_world, M_wc, K):
    vec = np.array([p_world.x, p_world.y, p_world.z, 1.0], dtype=np.float32)
    p_cam = M_wc @ vec
    z = p_cam[2]
    if z <= 0:
        return None, None, False
    p_img = K @ (p_cam[:3] / z)
    return float(p_img[0]), float(p_img[1]), True

def in_image(u, v, width, height):
    return (0 <= u < width) and (0 <= v < height)

def safe_destroy(actor, label=""):
    if actor is None:
        return
    try:
        if hasattr(actor, 'is_alive') and not actor.is_alive:
            return
        actor.destroy()
    except RuntimeError as e:
        if "not found" not in str(e):
            print(f"[WARN] Destroy {label} id={getattr(actor,'id','?')} failed: {e}")

def get_visible_walkers_simple(sensor_transform, walkers, max_distance):
    cam_loc = sensor_transform.location
    visible = []
    for w in walkers:
        loc = w.get_location()
        if cam_loc.distance(loc) <= max_distance:
            visible.append({"id": w.id, "x": loc.x, "y": loc.y, "z": loc.z})
    return visible

def enforce_boundary(center_loc, walkers, controllers):
    for w, ctrl in zip(walkers, controllers):
        loc = w.get_location()
        dx = loc.x - center_loc.x
        dy = loc.y - center_loc.y
        if (dx*dx + dy*dy) > (RELOCATE_RADIUS * RELOCATE_RADIUS):
            new_target = random_point_in_disk(center_loc, SPAWN_RADIUS * 0.9)
            ctrl.go_to_location(new_target)

def random_point_in_disk(center_loc, radius):
    r = radius * math.sqrt(random.random())
    theta = random.random() * 2.0 * math.pi
    return carla.Location(
        x=center_loc.x + r * math.cos(theta),
        y=center_loc.y + r * math.sin(theta),
        z=center_loc.z
    )

def draw_boundary(world, center_loc, radius, segments=48, life_time=60.0, color=carla.Color(0, 255, 0)):
    dbg = world.debug
    for k in range(segments):
        a1 = 2 * math.pi * k / segments
        a2 = 2 * math.pi * (k + 1) / segments
        p1 = center_loc + carla.Location(x=radius * math.cos(a1), y=radius * math.sin(a1), z=0.15)
        p2 = center_loc + carla.Location(x=radius * math.cos(a2), y=radius * math.sin(a2), z=0.15)
        dbg.draw_line(p1, p2, thickness=0.07, life_time=life_time, color=color)

# ================== SPAWNING ==================

def spawn_walkers_retries(world, blueprint_library, center_loc):
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    walkers = []
    controllers = []
    actor_list = []

    for i in range(NUM_WALKERS):
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        if bp.has_attribute('color'):
            vals = bp.get_attribute('color').recommended_values
            if vals:
                bp.set_attribute('color', random.choice(vals))
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', f"walker_{i}")

        walker = None
        for attempt in range(20):
            nav_loc = world.get_random_location_from_navigation()
            if not nav_loc:
                continue
            # Keep cluster roughly inside disk by projecting toward center
            direction = nav_loc - center_loc
            length = math.sqrt(direction.x**2 + direction.y**2)
            if length > SPAWN_RADIUS:
                scale = SPAWN_RADIUS / length
                nav_loc.x = center_loc.x + direction.x * scale
                nav_loc.y = center_loc.y + direction.y * scale
            nav_loc.z = center_loc.z + 0.1
            transform = carla.Transform(nav_loc, carla.Rotation(yaw=random.uniform(-180, 180)))
            walker = world.try_spawn_actor(bp, transform)
            if walker:
                break
        if not walker:
            print(f"[ERROR] Could not spawn walker {i} after retries.")
            continue

        ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if not ctrl:
            print(f"[ERROR] Failed to spawn controller for walker {walker.id}. Destroying walker.")
            safe_destroy(walker, "orphan_walker")
            continue

        walkers.append(walker)
        controllers.append(ctrl)
        actor_list.extend([walker, ctrl])

    world.tick()

    # Initialize controllers (after a tick)
    for ctrl in controllers:
        ctrl.start()
        ctrl.set_max_speed(WALKER_MAX_SPEED)
        tgt = world.get_random_location_from_navigation()
        if tgt:
            ctrl.go_to_location(tgt)

    return walkers, controllers, actor_list

def spawn_walkers_batch(client, world, blueprint_library, center_loc):
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    spawn_cmds = []
    walker_blueprints_used = []
    for i in range(NUM_WALKERS):
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        if bp.has_attribute('color'):
            vals = bp.get_attribute('color').recommended_values
            if vals:
                bp.set_attribute('color', random.choice(vals))
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', f"walker_{i}")

        nav_loc = world.get_random_location_from_navigation()
        if not nav_loc:
            continue
        direction = nav_loc - center_loc
        length = math.sqrt(direction.x**2 + direction.y**2)
        if length > SPAWN_RADIUS:
            scale = SPAWN_RADIUS / length
            nav_loc.x = center_loc.x + direction.x * scale
            nav_loc.y = center_loc.y + direction.y * scale
        nav_loc.z = center_loc.z + 0.1
        transform = carla.Transform(nav_loc, carla.Rotation(yaw=random.uniform(-180, 180)))
        spawn_cmds.append(carla.command.SpawnActor(bp, transform))
        walker_blueprints_used.append(bp)

    results = client.apply_batch_sync(spawn_cmds, True)
    walker_ids = []
    for idx, r in enumerate(results):
        if r.error:
            print(f"[BATCH][WALKER][FAIL] {idx} error: {r.error}")
        else:
            walker_ids.append(r.actor_id)

    # Controllers
    ctrl_cmds = [
        carla.command.SpawnActor(controller_bp, carla.Transform(), wid)
        for wid in walker_ids
    ]
    ctrl_results = client.apply_batch_sync(ctrl_cmds, True)
    controller_ids = []
    for idx, r in enumerate(ctrl_results):
        if r.error:
            print(f"[BATCH][CTRL][FAIL] {idx} error: {r.error}")
        else:
            controller_ids.append(r.actor_id)

    world.tick()

    walkers = world.get_actors(walker_ids)
    controllers = world.get_actors(controller_ids)

    for ctrl in controllers:
        ctrl.start()
        ctrl.set_max_speed(WALKER_MAX_SPEED)
        tgt = world.get_random_location_from_navigation()
        if tgt:
            ctrl.go_to_location(tgt)

    return list(walkers), list(controllers), list(walkers) + list(controllers)

# ================== MAIN ==================

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    if TOWN_NAME:
        if world.get_map().name != TOWN_NAME:
            print(f"[INFO] Loading map {TOWN_NAME}...")
            world = client.load_world(TOWN_NAME)
            # wait a few ticks so navmesh is ready
            for _ in range(10):
                world.tick()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # Pick valid center
    center_loc = None
    for _ in range(30):
        loc = world.get_random_location_from_navigation()
        if loc:
            center_loc = loc
            break
    if center_loc is None:
        raise RuntimeError("Could not find a navigation location for center.")
    center_loc.z += 0.05
    print(f"[INFO] Center location: ({center_loc.x:.2f}, {center_loc.y:.2f}, {center_loc.z:.2f})")

    walkers = []
    walker_controllers = []
    cameras = []
    actor_list = []
    image_queues = {}

    try:
        # ----- Spawn Walkers -----
        if USE_BATCH_SPAWN:
            walkers, walker_controllers, batch_list = spawn_walkers_batch(client, world, blueprint_library, center_loc)
            actor_list.extend(batch_list)
        else:
            walkers, walker_controllers, retry_list = spawn_walkers_retries(world, blueprint_library, center_loc)
            actor_list.extend(retry_list)

        if len(walkers) < NUM_WALKERS:
            print(f"[WARN] Spawned only {len(walkers)} / {NUM_WALKERS} walkers.")

        # ----- Cameras -----
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
        cam_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        cam_bp.set_attribute('fov', str(CAM_FOV_DEG))
        if cam_bp.has_attribute('enable_postprocess_effects'):
            cam_bp.set_attribute('enable_postprocess_effects', 'false')

        fixed_cam_transform = carla.Transform(
            carla.Location(x=center_loc.x, y=center_loc.y, z=CAM_HEIGHT),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )
        fixed_cam = world.spawn_actor(cam_bp, fixed_cam_transform)
        cameras.append(("fixed_camera", fixed_cam))
        actor_list.append(fixed_cam)

        if ENABLE_WALKER_CAMERAS:
            for i, wk in enumerate(walkers):
                cam_transform = carla.Transform(carla.Location(x=0, y=0, z=1.7))
                cam = world.spawn_actor(cam_bp, cam_transform, attach_to=wk)
                cameras.append((f"walker_{i}_camera", cam))
                actor_list.append(cam)

        for name, cam in cameras:
            q = queue.Queue()
            image_queues[cam.id] = q
            cam.listen(q.put)

        for name, _ in cameras:
            Path(f"{OUTPUT_DIR}/{name}").mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        K_overhead = build_camera_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_FOV_DEG)

        if DRAW_BOUNDARY:
            draw_boundary(world, center_loc, SPAWN_RADIUS)

        world.tick()  # prime sensors

        start_time = time.time()
        per_frame_times = []

        for frame in range(TOTAL_FRAMES):
            t0 = time.time()
            world.tick()
            snapshot = world.get_snapshot()
            world_frame_id = snapshot.frame

            if frame % BOUNDARY_CHECK_EVERY == 0:
                enforce_boundary(center_loc, walkers, walker_controllers)

            metadata_frame = {
                "frame_index": frame,
                "world_frame_id": world_frame_id,
                "sim_time": snapshot.timestamp.elapsed_seconds,
                "cameras": []
            }

            # Overhead transform & matrix
            overhead_transform = fixed_cam.get_transform()
            M_wc_overhead = world_to_camera_matrix(overhead_transform)

            for name, cam in cameras:
                q = image_queues[cam.id]
                img = q.get(timeout=5.0)
                while img.frame != world_frame_id:
                    img = q.get(timeout=5.0)
                img.save_to_disk(f"{OUTPUT_DIR}/{name}/{frame:05d}.png")

                cam_transform = cam.get_transform()
                simple_visible = get_visible_walkers_simple(
                    cam_transform, walkers, max_distance=SPAWN_RADIUS * 1.2
                )

                projection_data = []
                if WRITE_PROJECTION_DATA and name == "fixed_camera":
                    for w in walkers:
                        wloc = w.get_location()
                        u, v, front = project_world_point(wloc, M_wc_overhead, K_overhead)
                        if front and u is not None:
                            projection_data.append({
                                "walker_id": w.id,
                                "u": u,
                                "v": v,
                                "inside_image": in_image(u, v, IMAGE_WIDTH, IMAGE_HEIGHT)
                            })
                        else:
                            projection_data.append({
                                "walker_id": w.id,
                                "u": None,
                                "v": None,
                                "inside_image": False
                            })

                meta_cam = {
                    "name": name,
                    "camera_actor_id": cam.id,
                    "location": {
                        "x": cam_transform.location.x,
                        "y": cam_transform.location.y,
                        "z": cam_transform.location.z
                    },
                    "rotation": {
                        "pitch": cam_transform.rotation.pitch,
                        "yaw": cam_transform.rotation.yaw,
                        "roll": cam_transform.rotation.roll
                    },
                    "visible_walkers_simple": simple_visible
                }
                if projection_data:
                    meta_cam["projection_visibility"] = projection_data

                metadata_frame["cameras"].append(meta_cam)

            with open(f"{OUTPUT_DIR}/frame_{frame:05d}.json", "w") as f:
                json.dump(metadata_frame, f, indent=2)

            per_frame_times.append(time.time() - t0)

        total_time = time.time() - start_time
        avg_ms = (total_time / TOTAL_FRAMES) * 1000.0
        print(f"[INFO] Completed {TOTAL_FRAMES} frames in {total_time:.2f}s (avg {avg_ms:.2f} ms/frame).")

    finally:
        print("[INFO] Cleaning up...")
        for _, cam in cameras:
            try:
                cam.stop()
            except:
                pass
        # Controllers first
        for ctrl in walker_controllers:
            safe_destroy(ctrl, "controller")
        for wk in walkers:
            safe_destroy(wk, "walker")
        for name, cam in cameras:
            safe_destroy(cam, f"camera:{name}")
        for a in actor_list:
            safe_destroy(a, "actor_list")

        try:
            world.apply_settings(original_settings)
        except:
            pass
        print("[INFO] Cleanup complete.")

# Entry
if __name__ == "__main__":
    main()
