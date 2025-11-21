import os
import glob
import json
import numpy as np
import cv2
from scipy.optimize import minimize

# === User-configurable parameters ===
ann_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\Wildtrack_dataset_full\Wildtrack_dataset\annotations_positions"
Lambda_max = 10.0       # total capacity for sum of λ_i
FRAME_IDX = 200         # scheduler start frame index
WINDOW_SIZE = 100       # number of frames to compute average speed

# === 1. Hard-code camera rotation vectors (rvecs) for C1–C7 ===
rvecs = {
    'C1': np.array([ 1.759099006652832,  0.46710100769996643, -0.331699013710022]),
    'C2': np.array([ 0.6167870163917542, -2.14595890045166,    1.6577140092849731]),
    'C3': np.array([ 0.5511789917945862,  2.229501962661743,  -1.7721869945526123]),
    'C4': np.array([ 1.6647210121154785,  0.9668620228767395, -0.6937940120697021]),
    'C5': np.array([ 1.2132920026779175, -1.4771349430084229,  1.2775369882583618]),
    'C6': np.array([ 1.6907379627227783, -0.3968360126018524,  0.355197012424469]),
    'C7': np.array([ 1.6439390182495117,  1.126188039779663, -0.7273139953613281]),
}

# === 2. Compute each camera's optical-axis in world coords ===
axes = {}
cams = []
for cam_id, rvec in rvecs.items():
    R, _ = cv2.Rodrigues(rvec)
    cam_z = np.array([0.0, 0.0, 1.0])
    axis_world = R.T.dot(cam_z)
    axis_world /= np.linalg.norm(axis_world)
    axes[cam_id] = axis_world
    cams.append(cam_id)
    cams = sorted(cams)
C = len(cams)

# === 3. Compute average speeds per person over frames [FRAME_IDX-WINDOW_SIZE..FRAME_IDX] ===
def compute_average_speeds(folder, end_frame, window_size):
    start_frame = max(0, end_frame - window_size)
    tracks = {}
    for js in sorted(glob.glob(os.path.join(folder, '*.json'))):
        frame = int(os.path.basename(js).split('.')[0])
        if frame < start_frame or frame > end_frame:
            continue
        with open(js) as f:
            data = json.load(f)
        for obj in data:
            pid = obj.get('personID')
            pos_id = obj.get('positionID')
            if pid is None or pos_id is None:
                continue
            x_idx = pos_id % 480
            y_idx = pos_id // 480
            X = -3.0 + 0.025 * x_idx
            Y = -9.0 + 0.025 * y_idx
            coord = np.array([X, Y, 0.0])
            tracks.setdefault(pid, []).append((frame, coord))
    avg_speeds = {}
    for pid, pts in tracks.items():
        pts = sorted(pts, key=lambda x: x[0])
        if len(pts) < 2:
            avg_speeds[pid] = 0.0
            continue
        dists, times = [], []
        for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
            dt = f1 - f0
            if dt > 0:
                dists.append(np.linalg.norm(p1 - p0))
                times.append(dt)
        avg_speeds[pid] = sum(dists) / sum(times) if times else 0.0
    return avg_speeds

avg_speeds = compute_average_speeds(ann_folder, FRAME_IDX, WINDOW_SIZE)
persons = sorted(avg_speeds.keys())
P_all = len(persons)

# === 4. Load visibility at FRAME_IDX and filter persons ===
vis_mask = np.zeros((C, P_all), dtype=bool)
vis_file = os.path.join(ann_folder, f"{FRAME_IDX:08d}.json")
with open(vis_file) as f:
    vis_data = json.load(f)
for obj in vis_data:
    pid = obj.get('personID')
    if pid in persons:
        j = persons.index(pid)
        for v in obj.get('views', []):
            if v.get('xmin', -1) != -1:
                cam_idx = v.get('viewNum')
                vis_mask[cam_idx, j] = True
any_vis = vis_mask.any(axis=0)
persons = [pid for pid, ok in zip(persons, any_vis) if ok]
speeds = np.array([avg_speeds[pid] for pid in persons], dtype=float)
vis_mask = vis_mask[:, any_vis]
P = len(persons)
print(f"Filtered to {P} visible persons from {P_all} total.")

# === 5. Build g-vector (length = 4P) ===
g = np.repeat(speeds, 4)
print(f"g-vector (4P): {g}")

# === 6. Build A-matrix (C x 4P) and print rows as 4-column blocks ===
obs_dirs = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], dtype=float)
A_mat = np.zeros((C, 4*P), dtype=float)
for i, cam in enumerate(cams):
    d = axes[cam]
    for j_idx, pid in enumerate(persons):
        if not vis_mask[i, j_idx]:
            continue
        for k in range(4):
            cos_t = np.dot(d, obs_dirs[k])
            eta = 0.5 * (1 + cos_t)
            A_mat[i, 4*j_idx + k] = eta
print("A-matrix per camera with 4 columns per person:")
for i, cam in enumerate(cams):
    block = A_mat[i].reshape(P, 4)
    print(f"Camera {cam} (shape {block.shape}):")
    print(block)

# === 7. Define uniform service rates ===
mu = np.ones(C, dtype=float)

# === 8. CTMC steady-state calculation ===
def compute_steady_state(lambda_arrivals, mu):
    Lambda = np.sum(lambda_arrivals)
    A_val = np.sum(lambda_arrivals / (Lambda + mu))
    B_val = np.sum(lambda_arrivals / (mu * (Lambda + mu)))
    pi0 = (1 - A_val) / (1 + Lambda * B_val)
    pi_i0 = pi0 * (lambda_arrivals / ((Lambda + mu) * (1 - A_val)))
    pi_i1 = pi_i0 * (Lambda / mu)
    return pi0, pi_i0, pi_i1

# === 9. AoI objective ===
def compute_aoi_objective(lambda_arrivals, mu, A_mat, g):
    pi0, pi_i0, pi_i1 = compute_steady_state(lambda_arrivals, mu)
    phi = mu * (pi_i0 + pi_i1)
    Lambda = np.sum(lambda_arrivals)
    pi_tot = pi_i0 + pi_i1
    D = np.zeros_like(mu)
    for i in range(len(mu)):
        term1 = pi0 * lambda_arrivals[i] / mu[i]
        term2 = np.sum([
            pi_tot[k] * lambda_arrivals[i] * (mu[k] / (mu[k] + Lambda)) *
            (1/(mu[k] + Lambda) + 1/mu[i])
            for k in range(len(mu))
        ])
        D[i] = (term1 + term2) / phi[i]
    Phi = A_mat.T.dot(phi)
    num = A_mat.T.dot(phi * D)
    T_delay = num / Phi
    AoI = T_delay + 1 / Phi
    return np.dot(g, AoI)

# === 10. Optimize λ with SLSQP ===
x0 = np.ones(C) * (Lambda_max / C)
bounds = [(1e-6, None)] * C
cons = {'type':'ineq','fun': lambda x: Lambda_max - np.sum(x)}
res = minimize(lambda x: compute_aoi_objective(x, mu, A_mat, g), x0,
               method='SLSQP', bounds=bounds, constraints=cons)
lambda_opt = res.x
J_opt = res.fun

# === 11. Compare to equal-allocation ===
J_equal = compute_aoi_objective(x0, mu, A_mat, g)
print("Start frame:", FRAME_IDX)
print("Window size:", WINDOW_SIZE)
print("Optimal λ_i:", lambda_opt)
print("AoI (optimal):", J_opt)
print("Equal λ_i:", x0)
print("AoI (equal):", J_equal)
