import os
import glob
import json
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === Settings ===
data_folder   = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\output"
Lambda_max    = 10.0
EPS           = 1e-6
penalty_coef  = 1e3    # coefficient for linear penalty term in PSO

# === 1. Mapping walker idx → person ID ===
def walker_idx_to_pid(idx):
    return 24 + 2 * idx

# === 2. Average speed per pedestrian ===
def compute_average_speeds(folder):
    tracks = {}
    files = sorted(glob.glob(os.path.join(folder, 'frame_*.json')))
    for js in files:
        frame = int(os.path.basename(js).split('_')[1].split('.')[0])
        data  = json.load(open(js))
        for cam in data['cameras']:
            if cam['name']=='fixed_camera': continue
            pid = walker_idx_to_pid(int(cam['name'].split('_')[1]))
            xyz = cam['location']
            pos = np.array([xyz['x'], xyz['y'], xyz['z']])
            tracks.setdefault(pid, []).append((frame, pos))
    avg = {}
    for pid, pts in tracks.items():
        pts = sorted(pts)
        if len(pts) < 2:
            avg[pid] = 0.0
            continue
        dist, dt = 0.0, 0
        for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
            d = np.linalg.norm(p1 - p0)
            dist += d; dt += (f1 - f0)
        avg[pid] = dist / dt if dt > 0 else 0.0
    return avg

# === 3. Euler→vector helper ===
def euler_to_vector(pitch, yaw):
    p = math.radians(pitch); y = math.radians(yaw)
    return np.array([math.cos(p)*math.cos(y), math.cos(p)*math.sin(y), math.sin(p)])

# === 4. CTMC steady-state & AoI objective ===
def compute_steady_state(lam, mu):
    Λ = lam.sum()
    A = np.sum(lam / (Λ + mu)); B = np.sum(lam / (mu * (Λ + mu)))
    π0 = (1 - A) / (1 + Λ * B)
    πi0 = π0 * lam / ((Λ + mu) * (1 - A)); πi1 = πi0 * (Λ / mu)
    return π0, πi0, πi1


def compute_aoi_objective(lam, mu, A_mat, g):
    π0, πi0, πi1 = compute_steady_state(lam, mu)
    φ = mu * (πi0 + πi1); Λ = lam.sum(); πt = πi0 + πi1
    D = np.zeros_like(mu)
    for i in range(len(mu)):
        t1 = π0 * lam[i] / mu[i]
        t2 = sum(πt[k] * lam[i] * (mu[k] / (mu[k] + Λ)) *
                 (1/(mu[k] + Λ) + 1/mu[i]) for k in range(len(mu)))
        D[i] = (t1 + t2) / φ[i]
    mask = A_mat.sum(0) > 0
    A_vis = A_mat[:, mask]; g_vis = g[mask]
    Phi = A_vis.T.dot(φ); Num = A_vis.T.dot(φ * D)
    AoI = (Num / Phi) + 1.0 / Phi
    return float(g_vis.dot(AoI))

# === 5. Penalty-based PSO (linear penalty) returns top swarm solutions ===
def pso_optimize(obj, mu, A_mat, g,
                 swarm_size=30, max_iter=100,
                 w=0.5, c1=1.5, c2=1.5):
    N = len(mu)
    # initialize
    X = np.random.rand(swarm_size, N); X = np.clip(X, EPS, None)
    V = np.zeros_like(X)
    pbest = X.copy(); pval = np.full(swarm_size, np.inf)
    gbest_val = np.inf

    def penalized(lam):
        loss = obj(lam, mu, A_mat, g)
        cons = lam.sum() - Lambda_max
        return loss + penalty_coef * cons

    # evaluate initial swarm
    for i in range(swarm_size):
        v = penalized(X[i])
        pval[i] = v
        if v < gbest_val:
            gbest_val = v

    # PSO main loop
    for _ in range(max_iter):
        for i in range(swarm_size):
            r1, r2 = np.random.rand(N), np.random.rand(N)
            # velocity update (using personal bests implicitly via pbest)
            V[i] = (w * V[i]
                    + c1 * r1 * (pbest[i] - X[i])
                    + c2 * r2 * (X[np.argmin(pval)] - X[i]))
            X[i] += V[i]
            X[i] = np.clip(X[i], EPS, None)
            v = penalized(X[i])
            # update personal best
            if v < pval[i]:
                pval[i] = v; pbest[i] = X[i].copy()
                if v < gbest_val:
                    gbest_val = v
    # return all personal best solutions and values
    return pbest, pval

# === 6. Hybrid: one PSO → top-10 SLSQP refinements ===
def hybrid_opt(obj, mu, A_mat, g, top_k=10):
    # run PSO once to get candidate starts
    pbest, pval = pso_optimize(obj, mu, A_mat, g)
    # select top-k particle solutions
    idxs = np.argsort(pval)[:top_k]
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons = {'type':'eq', 'fun': lambda lam: lam.sum() - Lambda_max}
    best_lam, best_val = None, np.inf

    for idx in idxs:
        lam_init = pbest[idx]
        res = minimize(obj, lam_init, args=(mu, A_mat, g),
                       method='SLSQP', bounds=bounds,
                       constraints=[cons],
                       options={'ftol':1e-6, 'maxiter':50})
        lam_ref = res.x; val = obj(lam_ref, mu, A_mat, g)
        if val < best_val:
            best_val = val; best_lam = lam_ref.copy()
    return best_lam

# === 7. Main evaluation loop ===
avg_speeds = compute_average_speeds(data_folder)
avg_mean   = np.mean(list(avg_speeds.values()))

sample   = json.load(open(os.path.join(data_folder,'frame_00000.json')))
cams     = sorted(c['name'] for c in sample['cameras'])
num_cams = len(cams)
obs_dirs = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]])
files    = sorted(glob.glob(os.path.join(data_folder,'frame_*.json')))
frames   = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
mu       = np.random.uniform(0.5, 2.0, size=num_cams)

AoI_hybrid, AoI_avg = [], []
for fp, t in zip(files, frames):
    data = json.load(open(fp))
    axes = {c['name']: euler_to_vector(c['rotation']['pitch'], c['rotation']['yaw'])
            for c in data['cameras']}
    pids = sorted({pid for c in data['cameras'] for pid in c['visible_ids']})
    P_t  = len(pids)
    speeds = np.array([avg_speeds.get(pid, avg_mean) for pid in pids])
    g_t    = np.repeat(speeds, 4)
    vis    = np.zeros((num_cams, P_t), bool)
    for i,name in enumerate(cams):
        entry = next(c for c in data['cameras'] if c['name']==name)
        for pid in entry['visible_ids']:
            vis[i, pids.index(pid)] = True
    A_mat = np.zeros((num_cams, 4*P_t))
    for i,name in enumerate(cams):
        d = axes[name]
        for j in range(P_t):
            if not vis[i,j]: continue
            for k,ud in enumerate(obs_dirs):
                A_mat[i,4*j+k] = 0.5 * (1 + d.dot(ud))
    lam0  = np.ones(num_cams) * (Lambda_max/num_cams)
    lam_h = hybrid_opt(compute_aoi_objective, mu, A_mat, g_t, top_k=10)
    AoI_h  = compute_aoi_objective(lam_h, mu, A_mat, g_t)
    AoI_u  = compute_aoi_objective(lam0, mu, A_mat, g_t)
    AoI_hybrid.append(AoI_h)
    AoI_avg.append(AoI_u)

# === 8. Results & plot ===
print("Frames:", frames)
print("Hybrid AoI:", AoI_hybrid)
print("Uniform AoI:", AoI_avg)
plt.figure()
plt.plot(frames, AoI_hybrid, label='Hybrid PSO+SLSQP AoI')
plt.plot(frames, AoI_avg, label='Uniform AoI')
plt.xlabel('Frame')
plt.ylabel('AoI')
plt.legend()
plt.show()
