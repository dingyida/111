import carla
import random
import time
import json
import math
import queue
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ================ CONFIG ================
TOTAL_FRAMES           = 200
FPS                    = 60
OUTPUT_DIR             = "output"
NUM_WALKERS            = 7    # ensure exactly 7 pedestrians

# Walker spawn disk
SPAWN_RADIUS           = 10.0
RELOCATE_RADIUS        = 11.5
USE_DISK_TARGETS       = True
MIN_SEP_DIST           = 1.2
SEPARATION_PASSES      = 3

# Camera settings
USE_CUSTOM_CAMERA_POSE = True
CAM_FOV_DEG            = 90.0
IMAGE_WIDTH            = 1280
IMAGE_HEIGHT           = 720
CAM_LOC                = carla.Location(x=-85.54, y=143.20, z=8.29)
CAM_ROT                = carla.Rotation(pitch=-43.14, yaw=-171.19, roll=0.00)
# Fallback overhead
CAM_HEIGHT             = 25.0

# Where to spawn walkers
WALKER_CENTER          = carla.Location(x=-98.08, y=141.11, z=1.09)

# Walker AI
ENABLE_WALKER_CAMERAS  = True
WALKER_MAX_SPEED       = 1.3
BOUNDARY_CHECK_EVERY   = 5

# Visual
DRAW_BOUNDARY          = True
WEATHER_PARAMS = carla.WeatherParameters(
    cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
    wind_intensity=0.0, sun_azimuth_angle=135.0, sun_altitude_angle=70.0,
    fog_density=0.0, fog_distance=0.0, wetness=0.0
)

# ========== CAMERA / PROJECTION HELPERS ==========

def build_camera_intrinsics(w, h, fov):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0,   1]], dtype=np.float32)

def world_to_camera_matrix(tf):
    M = tf.get_inverse_matrix()
    return np.array(M, dtype=np.float32).reshape(4, 4)

def project_world_point(p, M_wc, K):
    """
    Corrected for CARLA camera axes:
      - X_cam is forward (depth)
      - Y_cam is right → image u
      - Z_cam is up    → image v (with flip)
    """
    vec   = np.array([p.x, p.y, p.z, 1.0], dtype=np.float32)
    p_cam = M_wc @ vec

    depth = p_cam[0]              # forward along +X
    if depth <= 0:
        return None, None, False

    # pinhole projection: u = f*(Y_cam/depth) + cx; v = -f*(Z_cam/depth) + cy
    u = float(K[0,0] * (p_cam[1] / depth) + K[0,2])
    v = float(K[1,1] * (-p_cam[2] / depth) + K[1,2])
    return u, v, True

def in_image(u, v, w, h):
    return 0 <= u < w and 0 <= v < h

# ============ ACTOR MANAGEMENT ============

def safe_destroy(a, label=""):
    if a is None:
        return
    try:
        if hasattr(a, 'is_alive') and not a.is_alive:
            return
        a.destroy()
    except RuntimeError as e:
        if "not found" not in str(e):
            print(f"[WARN] Destroy {label} id={getattr(a, 'id', '?')} failed: {e}")

# ============ GEOMETRY / BOUNDARY ============

def random_point_in_disk(center, radius):
    r = radius * math.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    return carla.Location(
        x = center.x + r * math.cos(theta),
        y = center.y + r * math.sin(theta),
        z = center.z
    )

def enforce_boundary(center, walkers, controllers):
    for w, ctrl in zip(walkers, controllers):
        loc = w.get_location()
        dx, dy = loc.x - center.x, loc.y - center.y
        if dx*dx + dy*dy > RELOCATE_RADIUS**2:
            if USE_DISK_TARGETS:
                tgt = random_point_in_disk(center, SPAWN_RADIUS * 0.95)
            else:
                nav = w.get_world().get_random_location_from_navigation()
                tgt = nav if nav else random_point_in_disk(center, SPAWN_RADIUS)
            ctrl.go_to_location(tgt)

def draw_boundary(world, center, radius):
    dbg = world.debug
    pts = 64
    for i in range(pts):
        a1 = 2 * math.pi * i / pts
        a2 = 2 * math.pi * (i + 1) / pts
        p1 = center + carla.Location(x=radius * math.cos(a1), y=radius * math.sin(a1), z=0.15)
        p2 = center + carla.Location(x=radius * math.cos(a2), y=radius * math.sin(a2), z=0.15)
        dbg.draw_line(p1, p2, thickness=0.05, life_time=120.0, color=carla.Color(0,255,0))

def repel_positions(pos_list, min_dist):
    for _ in range(SEPARATION_PASSES):
        for i in range(len(pos_list)):
            for j in range(i+1, len(pos_list)):
                dx = pos_list[j].x - pos_list[i].x
                dy = pos_list[j].y - pos_list[i].y
                d2 = dx*dx + dy*dy
                if d2 < 1e-9:
                    continue
                if d2 < (min_dist * min_dist):
                    d = math.sqrt(d2)
                    overlap = (min_dist - d) / 2.0
                    if d > 0:
                        ux, uy = dx/d, dy/d
                        pos_list[i].x -= ux * overlap
                        pos_list[i].y -= uy * overlap
                        pos_list[j].x += ux * overlap
                        pos_list[j].y += uy * overlap

# ============ IMAGE LABELING ============

def annotate_image_overwrite(path, labels):
    if not labels:
        return
    try:
        img = Image.open(path).convert("RGB")
    except:
        print(f"[WARN] cannot open {path}")
        return
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for L in labels:
        x, y = int(L['u']), int(L['v'])
        txt = str(L['id'])
        pad = 2

        # get text width & height in a version-safe way
        try:
            tw, th = font.getsize(txt)
        except AttributeError:
            # fallback to textbbox (Pillow ≥8.0)
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # draw a little black box behind the text
        draw.rectangle([x, y - th, x + tw + pad, y + pad], fill=(0, 0, 0))
        draw.text((x, y - th), txt, fill=(255, 255, 0), font=font)

    img.save(path)


# =========== SPAWN WALKERS ===========

def spawn_walkers(world, blueprints, center):
    walker_bps = blueprints.filter('walker.pedestrian.*')
    ctrl_bp    = blueprints.find('controller.ai.walker')
    walkers, ctrls, actors = [], [], []

    pts = [random_point_in_disk(center, SPAWN_RADIUS * 0.98) for _ in range(NUM_WALKERS)]
    repel_positions(pts, MIN_SEP_DIST)

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

        loc = pts[i]
        loc.z = center.z + 0.15
        tf = carla.Transform(loc, carla.Rotation(yaw=random.uniform(-180,180)))
        w = world.try_spawn_actor(bp, tf)
        if not w:
            # fallback to nav-mesh
            for _ in range(15):
                nav = world.get_random_location_from_navigation()
                if not nav: continue
                d = math.hypot(nav.x-center.x, nav.y-center.y)
                if d > SPAWN_RADIUS:
                    scale = SPAWN_RADIUS / d
                    nav.x = center.x + (nav.x-center.x)*scale
                    nav.y = center.y + (nav.y-center.y)*scale
                nav.z = center.z + 0.15
                w = world.try_spawn_actor(bp, carla.Transform(nav, carla.Rotation(yaw=random.uniform(-180,180))))
                if w:
                    break
        if not w:
            print(f"[ERROR] Could not spawn walker {i}.")
            continue

        c = world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        if not c:
            print(f"[ERROR] no controller for walker {w.id}.")
            safe_destroy(w)
            continue

        walkers.append(w)
        ctrls.append(c)
        actors += [w, c]

    world.tick()
    for c in ctrls:
        c.start()
        c.set_max_speed(WALKER_MAX_SPEED)
        tgt = ( random_point_in_disk(center, SPAWN_RADIUS * 0.95)
                if USE_DISK_TARGETS
                else world.get_random_location_from_navigation() )
        if tgt:
            c.go_to_location(tgt)

    print("[INFO] Walker spawn locations:")
    for idx, w in enumerate(walkers):
        l = w.get_location()
        print(f"  Walker {idx} (id={w.id}) at ({l.x:.2f},{l.y:.2f},{l.z:.2f})")

    return walkers, ctrls, actors

# ============ MAIN ============

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # cleanup existing
    for a in world.get_actors().filter('walker.pedestrian.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.ai.walker'):
        a.destroy()

    orig = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode    = True
    s.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(s)
    world.set_weather(WEATHER_PARAMS)

    bp_lib = world.get_blueprint_library()
    center = WALKER_CENTER

    walkers, ctrls, actor_list = spawn_walkers(world, bp_lib, center)

    # camera blueprint
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
    cam_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
    cam_bp.set_attribute('fov',           str(CAM_FOV_DEG))
    if cam_bp.has_attribute('enable_postprocess_effects'):
        cam_bp.set_attribute('enable_postprocess_effects', 'false')

    cameras, queues = [], {}
    # fixed camera
    tf = (carla.Transform(CAM_LOC, CAM_ROT)
          if USE_CUSTOM_CAMERA_POSE
          else carla.Transform(
                carla.Location(x=center.x, y=center.y, z=CAM_HEIGHT),
                carla.Rotation(pitch=-90, yaw=0, roll=0)))
    fixed_cam = world.spawn_actor(cam_bp, tf)
    cameras.append(("fixed_camera", fixed_cam))
    actor_list.append(fixed_cam)

    # walker cameras
    if ENABLE_WALKER_CAMERAS:
        for i, w in enumerate(walkers):
            tf2 = carla.Transform(carla.Location(0,0,1.7))
            c = world.spawn_actor(cam_bp, tf2, attach_to=w)
            cameras.append((f"walker_{i}_camera", c))
            actor_list.append(c)

    # prepare queues & dirs
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for name, cam in cameras:
        q = queue.Queue()
        queues[cam.id] = q
        cam.listen(q.put)
        Path(f"{OUTPUT_DIR}/{name}").mkdir(parents=True, exist_ok=True)

    K = build_camera_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_FOV_DEG)
    if DRAW_BOUNDARY:
        draw_boundary(world, center, SPAWN_RADIUS)
    world.tick()
    start = time.time()

    for frame in range(TOTAL_FRAMES):
        world.tick()
        snap = world.get_snapshot()

        if frame % BOUNDARY_CHECK_EVERY == 0:
            enforce_boundary(center, walkers, ctrls)

        cams_json = []
        for name, cam in cameras:
            img = queues[cam.id].get(timeout=5.0)
            while img.frame != snap.frame:
                img = queues[cam.id].get(timeout=5.0)

            path_img = f"{OUTPUT_DIR}/{name}/{frame:05d}.png"
            img.save_to_disk(path_img)

            tf_cam = cam.get_transform()
            M_wc  = world_to_camera_matrix(tf_cam)
            vis_ids, labels = [], []
            for w in walkers:
                u, v, ok = project_world_point(w.get_location(), M_wc, K)
                if ok and in_image(u, v, IMAGE_WIDTH, IMAGE_HEIGHT):
                    vis_ids.append(w.id)
                    labels.append({"id": w.id, "u": u, "v": v})

            annotate_image_overwrite(path_img, labels)

            loc, rot = tf_cam.location, tf_cam.rotation
            cams_json.append({
                "name": name,
                "location": {"x": loc.x, "y": loc.y, "z": loc.z},
                "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
                "visible_ids": vis_ids
            })

        with open(f"{OUTPUT_DIR}/frame_{frame:05d}.json", "w") as jf:
            json.dump({"cameras": cams_json}, jf, indent=2)

    total = time.time() - start
    print(f"[INFO] Completed {TOTAL_FRAMES} frames in {total:.2f}s "
          f"(avg {total/TOTAL_FRAMES*1000:.1f} ms/frame)")

    # cleanup
    print("[INFO] Cleaning up…")
    for _, cam in cameras:
        try: cam.stop()
        except: pass
    for c in ctrls:    safe_destroy(c, "controller")
    for w in walkers: safe_destroy(w, "walker")
    for _, cam in cameras: safe_destroy(cam, "camera")
    for a in actor_list: safe_destroy(a, "actor_list")
    try:
        world.apply_settings(orig)
    except:
        pass
    print("[INFO] Cleanup complete.")

if __name__ == "__main__":
    main()
