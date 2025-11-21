import os
import glob
import json
import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# === Settings ===
ann_folder   = r"C:\Users\dyd\PycharmProjects\PythonProject\Wildtrack_dataset_full\Wildtrack_dataset\annotations_positions"
Lambda_max   = 10.0
FRAME_START  = 200
NUM_FRAMES   = 20
FRAME_STEP   = 5
WINDOW_SIZE  = 100
EPS          = 1e-6

# === 1. Camera rotations & optical axes ===
rvecs = {
    'C1': np.array([ 1.759099,  0.467101, -0.331699]),
    'C2': np.array([ 0.616787, -2.145959,  1.657714]),
    'C3': np.array([ 0.551179,  2.229502, -1.772187]),
    'C4': np.array([ 1.664721,  0.966862, -0.693794]),
    'C5': np.array([ 1.213292, -1.477135,  1.277537]),
    'C6': np.array([ 1.690738, -0.396836,  0.355197]),
    'C7': np.array([ 1.643939,  1.126189, -0.727314]),
}
cams = sorted(rvecs.keys())
axes = {}
for cam in cams:
    R, _       = cv2.Rodrigues(rvecs[cam])
    axis_world = R.T.dot(np.array([0,0,1], float))
    axes[cam]  = axis_world / np.linalg.norm(axis_world)

# === 2. Average speed estimation ===
def compute_average_speeds(folder, end_frame, window_size):
    start = max(0, end_frame - window_size)
    tracks = {}
    for js in sorted(glob.glob(os.path.join(folder, '*.json'))):
        frame = int(os.path.basename(js).split('.')[0])
        if frame < start or frame > end_frame:
            continue
        data = json.load(open(js))
        for obj in data:
            pid = obj.get('personID')
            pos = obj.get('positionID')
            if pid is None or pos is None:
                continue
            x = -3.0 + 0.025 * (pos % 480)
            y = -9.0 + 0.025 * (pos // 480)
            tracks.setdefault(pid, []).append((frame, np.array([x,y,0.0])))
    avg_speeds = {}
    for pid, pts in tracks.items():
        pts = sorted(pts, key=lambda t: t[0])
        if len(pts) < 2:
            avg_speeds[pid] = 0.0
            continue
        dists, times = [], []
        for (f0,p0),(f1,p1) in zip(pts, pts[1:]):
            dt = f1 - f0
            if dt > 0:
                dists.append(np.linalg.norm(p1-p0))
                times.append(dt)
        avg_speeds[pid] = sum(dists)/sum(times) if times else 0.0
    return avg_speeds

avg_speeds    = compute_average_speeds(ann_folder, FRAME_START, WINDOW_SIZE)
avg_speed_avg = np.mean(list(avg_speeds.values()))

# === 3. CTMC & AoI helpers ===
def compute_steady_state(lam, mu):
    Λ     = lam.sum()
    A_val = np.sum(lam/(Λ + mu))
    B_val = np.sum(lam/(mu * (Λ + mu)))
    π0    = (1 - A_val) / (1 + Λ * B_val)
    πi0   = π0 * lam/((Λ + mu)*(1 - A_val))
    πi1   = πi0 * (Λ/mu)
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
            π_tot[k] * lam[i] * (mu[k]/(mu[k]+Λ)) *
            (1/(mu[k]+Λ) + 1/mu[i])
            for k in range(len(mu))
        ])
        D[i] = (term1 + term2) / φ[i]

    col_vis = (A_mat.sum(axis=0) > 0)
    A_vis   = A_mat[:, col_vis]
    g_vis   = g[col_vis]

    φD   = φ * D
    Phi  = A_vis.T.dot(φ)
    Num  = A_vis.T.dot(φD)
    T_d  = Num / Phi
    AoI  = T_d + 1.0/Phi
    return float(np.dot(g_vis, AoI))

# === 4. Random μ vector and observation dirs ===
mu       = np.random.uniform(0.5, 2.0, size=len(cams))
obs_dirs = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], float)

# === 5. Loop to collect AoI data ===
frames   = []
AoI_opt  = []
AoI_avg  = []
lam_avg  = np.ones(len(cams)) * (Lambda_max / len(cams))

for m in range(NUM_FRAMES):
    t = FRAME_START + m * FRAME_STEP
    frames.append(t)

    # Load detections
    data  = json.load(open(os.path.join(ann_folder, f"{t:08d}.json")))
    pids  = sorted({obj['personID'] for obj in data})
    speeds_t = np.array([avg_speeds.get(pid, avg_speed_avg) for pid in pids])
    g_t   = np.repeat(speeds_t, 4)
    P_t   = len(pids)

    # Visibility mask
    vis_mask = np.zeros((len(cams), P_t), bool)
    for obj in data:
        pid = obj['personID']
        j   = pids.index(pid)
        for v in obj.get('views', []):
            if v.get('xmin', -1) >= 0:
                vis_mask[v['viewNum'], j] = True

    # Build A_mat_t
    A_mat_t = np.zeros((len(cams), 4*P_t), float)
    for i, cam in enumerate(cams):
        d = axes[cam]
        for j in range(P_t):
            if not vis_mask[i, j]:
                continue
            for k, ud in enumerate(obs_dirs):
                A_mat_t[i, 4*j + k] = 0.5 * (1 + d.dot(ud))

    # Optimize λ
    bounds = [(EPS, Lambda_max)] * len(cams)
    sum_constraint = {'type':'eq', 'fun': lambda lam: np.sum(lam) - Lambda_max}
    res = minimize(
        fun        = compute_aoi_objective,
        x0         = lam_avg,
        args       = (mu, A_mat_t, g_t),
        method     = 'SLSQP',
        bounds     = bounds,
        constraints= [sum_constraint],
        options    = {'ftol':1e-6, 'maxiter':100}
    )
    lam_opt = res.x
    J_opt   = compute_aoi_objective(lam_opt, mu, A_mat_t, g_t)
    J_avg   = compute_aoi_objective(lam_avg, mu, A_mat_t, g_t)

    AoI_opt.append(J_opt)
    AoI_avg.append(J_avg)

# Output the AoI values
print("Frame indices:", frames)
print("Optimized AoI:   ", AoI_opt)
print("Average λ AoI:   ", AoI_avg)

# Plotting both series
plt.figure()
plt.plot(frames, AoI_opt, label='Optimized AoI')
plt.plot(frames, AoI_avg, label='Average λ AoI')
plt.xlabel('Frame Index')
plt.ylabel('AoI')
plt.legend()
plt.show()

