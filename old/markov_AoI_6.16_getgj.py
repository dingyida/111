import os
import json
import glob
import numpy as np
from scipy.optimize import minimize


# === 1. Load ground‐plane positions for each positionID ===
def load_positions(pos_txt_path):
    """
    Expects a text file with lines: <positionID> <X> <Y> <Z>
    """
    positions = {}
    with open(pos_txt_path, 'r') as f:
        for line in f:
            pid, x, y, z = line.split()
            positions[int(pid)] = np.array([float(x), float(y), float(z)])
    return positions


# === 2. Compute average speed per person over frames 0–100 ===
def compute_average_speeds(ann_folder, positions, max_frame=100):
    """
    - ann_folder: folder containing JSONs named '00000000.json', '00000005.json', ...
    - positions: dict mapping positionID -> 3D coords.
    - max_frame: highest frame index to include
    """
    # map personID -> list of (frame_idx, XYZ)
    tracks = {}
    for js_path in sorted(glob.glob(os.path.join(ann_folder, "*.json"))):
        frame_idx = int(os.path.basename(js_path).split('.')[0])
        if frame_idx > max_frame:
            break
        data = json.load(open(js_path))
        for obj in data:
            pid = obj['personID']
            pos_id = obj['positionID']
            if pos_id not in positions:
                continue
            coords = positions[pos_id]
            tracks.setdefault(pid, []).append((frame_idx, coords))

    # compute average speed = total distance / total time
    avg_speeds = {}
    for pid, pts in tracks.items():
        if len(pts) < 2:
            avg_speeds[pid] = 0.0
            continue
        # sort by frame
        pts = sorted(pts, key=lambda x: x[0])
        dists = []
        times = []
        for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
            dt = (f1 - f0)  # assume unit time per frame; you can scale by framerate
            d = np.linalg.norm(p1 - p0)
            dists.append(d)
            times.append(dt)
        avg_speeds[pid] = sum(dists) / sum(times)
    return avg_speeds


# === 3. Build g-vector (length = 4 * P) from speeds ===
def build_g_vector(avg_speeds, persons):
    # speeds in same order as `persons`
    speeds = np.array([avg_speeds.get(pid, 0.0) for pid in persons])
    # repeat each person’s speed for their 4 faces
    g = np.repeat(speeds, 4)
    return g


# === 4. Example: integrate with your previous scheduler code ===
# (Assumes you already have `build_A_matrix` and `compute_aoi_objective` defined)

# Paths (adjust as needed)
positions_txt = r"C:\Users\dyd\PycharmProjects\PythonProject\Wildtrack_dataset_full\Wildtrack_dataset\positions.txt"
ann_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\Wildtrack_dataset_full\Wildtrack_dataset\annotations_positions"

# Load data
positions = load_positions(positions_txt)
avg_speeds = compute_average_speeds(ann_folder, positions, max_frame=100)
persons = sorted(avg_speeds.keys())

# Build g
g_vector = build_g_vector(avg_speeds, persons)

# Build A once (as before, with 4P columns)
# A_mat, cams = build_A_matrix(axes, persons, positions)

# Optimize λ using SLSQP (same as before)
# x0 = np.ones(len(cams)) * (Lambda_max / len(cams))
# result = minimize(lambda x: compute_aoi_objective(x, mu, A_mat, g_vector),
#                   x0, method='SLSQP', bounds=bounds, constraints=constraints)
# lambda_opt = result.x

# === 5. (Optional) Run scheduler on frames 105–200 ===
# You can now use `lambda_opt` to drive a discrete‐time or event‐based
# simulation over frames 105–200, collecting empirical AoI metrics.

