# -*- coding: utf-8 -*-
import os
import glob
import json
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === Settings ===
data_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\output"
Lambda_max  = 10.0
EPS         = 1e-6

# === 1. Mapping from walker index to personID ===
def walker_idx_to_pid(idx):
    # 0→24, 1→26, 2→28, ..., 6→36
    return 24 + 2 * idx

# === 2. Compute average speeds from all frames ===
def compute_average_speeds(folder):
    tracks = {}
    json_files = sorted(glob.glob(os.path.join(folder, 'frame_*.json')))
    for js in json_files:
        frame_num = int(os.path.basename(js).split('_')[1].split('.')[0])
        data = json.load(open(js))
        for cam in data['cameras']:
            name = cam['name']
            if name == 'fixed_camera':
                continue
            idx = int(name.split('_')[1])
            pid = walker_idx_to_pid(idx)
            loc = cam['location']
            pos = np.array([loc['x'], loc['y'], loc['z']])
            tracks.setdefault(pid, []).append((frame_num, pos))
    avg_speeds = {}
    for pid, pts in tracks.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        if len(pts_sorted) < 2:
            avg_speeds[pid] = 0.0
            continue
        dists, times = [], []
        for (f0, p0), (f1, p1) in zip(pts_sorted, pts_sorted[1:]):
            dt = f1 - f0
            if dt > 0:
                dists.append(np.linalg.norm(p1 - p0))
                times.append(dt)
        avg_speeds[pid] = sum(dists) / sum(times) if times else 0.0
    return avg_speeds

avg_speeds    = compute_average_speeds(data_folder)
avg_speed_avg = np.mean(list(avg_speeds.values()))

# === 3. Euler → optical-axis helper ===
# Using user-provided spherical conversion (ignoring roll)

def euler_to_vector(pitch_deg, yaw_deg):
    pitch = math.radians(pitch_deg)
    yaw   = math.radians(yaw_deg)
    x = math.cos(pitch) * math.cos(yaw)
    y = math.cos(pitch) * math.sin(yaw)
    z = math.sin(pitch)
    return np.array([x, y, z])

# === 4. CTMC & AoI functions ===
def compute_steady_state(lam, mu):
    Λ     = lam.sum()
    A_val = np.sum(lam / (Λ + mu))
    B_val = np.sum(lam / (mu * (Λ + mu)))
    π0    = (1 - A_val) / (1 + Λ * B_val)
    πi0   = π0 * lam / ((Λ + mu) * (1 - A_val))
    πi1   = πi0 * (Λ / mu)
    return π0, πi0, πi1


def compute_aoi_objective(lam, mu, A_mat, g):
    π0, πi0, πi1 = compute_steady_state(lam, mu)
    φ            = mu * (πi0 + πi1)
    Λ            = lam.sum()
    π_tot        = πi0 + πi1
    D = np.zeros_like(mu)
    for i in range(len(mu)):
        term1 = π0 * lam[i] / mu[i]
        term2 = np.sum([
            π_tot[k] * lam[i] * (mu[k] / (mu[k] + Λ)) *
            (1 / (mu[k] + Λ) + 1 / mu[i])
            for k in range(len(mu))
        ])
        D[i] = (term1 + term2) / φ[i]
    col_vis = (A_mat.sum(axis=0) > 0)
    A_vis   = A_mat[:, col_vis]
    g_vis   = g[col_vis]
    φD      = φ * D
    Phi     = A_vis.T.dot(φ)
    Num     = A_vis.T.dot(φD)
    T_d     = Num / Phi
    AoI     = T_d + 1.0 / Phi
    return float(np.dot(g_vis, AoI))

# === 5. Setup & main loop ===
sample       = json.load(open(os.path.join(data_folder, 'frame_00000.json')))
cams         = sorted([cam['name'] for cam in sample['cameras']])
num_cams     = len(cams)
obs_dirs     = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], float)
json_files   = sorted(glob.glob(os.path.join(data_folder, 'frame_*.json')))
frames       = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in json_files]
lam_avg      = np.ones(num_cams) * (Lambda_max / num_cams)
mu           = np.random.uniform(0.5, 2.0, size=num_cams)
AoI_opt, AoI_avg = [], []

for fpath, t in zip(json_files, frames):
    data = json.load(open(fpath))
    # compute per-camera optical axes using new converter
    axes = {c['name']: euler_to_vector(c['rotation']['pitch'], c['rotation']['yaw'])
            for c in data['cameras']}
    all_pids = set()
    for c in data['cameras']:
        all_pids.update(c['visible_ids'])
    pids = sorted(all_pids)
    P_t  = len(pids)
    speeds_t = np.array([avg_speeds.get(pid, avg_speed_avg) for pid in pids])
    g_t      = np.repeat(speeds_t, 4)
    vis_mask = np.zeros((num_cams, P_t), bool)
    for i, name in enumerate(cams):
        entry = next(c for c in data['cameras'] if c['name']==name)
        for pid in entry['visible_ids']:
            j = pids.index(pid)
            vis_mask[i, j] = True
    A_mat_t = np.zeros((num_cams, 4*P_t), float)
    for i, name in enumerate(cams):
        d = axes[name]
        for j in range(P_t):
            if not vis_mask[i, j]: continue
            for k, ud in enumerate(obs_dirs):
                A_mat_t[i, 4*j + k] = 0.5 * (1 + d.dot(ud))
    # optimize λ
    bounds = [(EPS, Lambda_max)] * num_cams
    cons   = {'type':'eq', 'fun': lambda lam: np.sum(lam) - Lambda_max}
    res    = minimize(compute_aoi_objective, lam_avg,
                      args=(mu, A_mat_t, g_t),
                      method='SLSQP', bounds=bounds,
                      constraints=[cons],
                      options={'ftol':1e-6, 'maxiter':100})
    lam_opt = res.x
    AoI_opt.append(compute_aoi_objective(lam_opt, mu, A_mat_t, g_t))
    AoI_avg.append(compute_aoi_objective(lam_avg, mu, A_mat_t, g_t))

# 6. Output & plot
print("Frame indices:", frames)
print("Optimized AoI:   ", AoI_opt)
print("Average λ AoI:   ", AoI_avg)
plt.figure()
plt.plot(frames, AoI_opt, label='Optimized AoI')
plt.plot(frames, AoI_avg, label='Average λ AoI')
plt.xlabel('Frame Index')
plt.ylabel('AoI')
plt.legend()
plt.show()
