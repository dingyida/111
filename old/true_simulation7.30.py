import os
import glob
import json
import math
import heapq
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ========== Helpers ==========
def euler_to_vector(pitch_deg, yaw_deg):
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    return np.array([math.cos(p)*math.cos(y),
                     math.cos(p)*math.sin(y),
                     math.sin(p)])

def compute_steady_state(lam, mu):
    L = lam.sum()
    A = np.sum(lam / (L + mu))
    B = np.sum(lam / (mu * (L + mu)))
    pi0 = (1 - A) / (1 + L * B)
    pii0 = pi0 * lam / ((L + mu) * (1 - A))
    pii1 = pii0 * (L / mu)
    return pi0, pii0, pii1

def compute_aoi_objective(lam, mu, O_mat, g):
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    phi = mu * (pii0 + pii1) + 1e-6
    mask = O_mat.sum(axis=0) > 0
    O_vis = O_mat[:, mask]
    g_vis = g[mask]
    Phi = O_vis.T.dot(phi)
    L = lam.sum()
    busy_frac = (pii0 + pii1)
    vis_bin = (O_vis > 0).astype(float)
    P = vis_bin.T.dot(busy_frac)
    AoIvec = (1.0 / L) + (1.0 + 2.0 * P) / (Phi + 1e-6)
    return float(g_vis.dot(AoIvec))

def gradient_lambda(lam, mu, O_mat, g, eps=1e-8):
    base = compute_aoi_objective(lam, mu, O_mat, g)
    grad = np.zeros_like(lam)
    for i in range(len(lam)):
        step = eps * max(1.0, abs(lam[i]))
        lam2 = lam.copy(); lam2[i] += step
        grad[i] = (compute_aoi_objective(lam2, mu, O_mat, g) - base) / step
    return grad

def pso_optimize(obj, mu, O_mat, g, swarm_size, max_iter, w=0.5, c1=1.5, c2=1.5):
    N = len(mu)
    X = np.clip(np.random.rand(swarm_size, N), 1e-6, None)
    V = np.zeros_like(X)
    pbest = X.copy()
    pval = np.full(swarm_size, np.inf)
    def penalized(lam):
        loss = obj(lam, mu, O_mat, g)
        cons = max(0.0, lam.sum() - 20.0)
        return loss + 1e3 * cons
    for i in range(swarm_size):
        pval[i] = penalized(X[i])
    for _ in range(max_iter):
        gbest = pbest[np.argmin(pval)]
        for i in range(swarm_size):
            r1, r2 = np.random.rand(N), np.random.rand(N)
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(gbest-X[i])
            X[i] = np.clip(X[i] + V[i], 1e-6, None)
            val = penalized(X[i])
            if val < pval[i]:
                pval[i], pbest[i] = val, X[i].copy()
    return pbest, pval

def hybrid_opt(obj, mu, O_mat, g, top_k):
    pbest, pval = pso_optimize(obj, mu, O_mat, g, swarm_size=30, max_iter=100)
    idxs = np.argsort(pval)[:top_k]
    bounds = [(1e-6, 20.0)] * len(mu)
    cons = {'type': 'eq', 'fun': lambda lam: lam.sum() - 20.0}
    best, val = None, np.inf
    for i in idxs:
        res = minimize(obj, pbest[i], args=(mu, O_mat, g),
                       method='SLSQP', bounds=bounds,
                       constraints=[cons],
                       options={'ftol':1e-6, 'maxiter':100})
        if res.fun < val:
            best, val = res.x.copy(), res.fun
    return best

# ========== Build system parameters ==========
data_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\7walker"
files = sorted(glob.glob(os.path.join(data_folder, 'frame_*.json')))
sample = json.load(open(files[0]))
cams = sorted(c['name'] for c in sample['cameras'])
num_cams = len(cams)

# Service rates μ_i
np.random.seed(42)
mu = np.random.uniform(0.5, 2.0, size=num_cams)

# All object IDs
all_pids = sorted({
    pid
    for fp in files
    for c in json.load(open(fp))['cameras']
    for pid in c.get('visible_ids', [])
})
num_objs = len(all_pids)

# Visibility & O_mat at frame 0
axes = {c['name']: euler_to_vector(c['rotation']['pitch'], c['rotation']['yaw'])
        for c in sample['cameras']}
vis = np.zeros((num_cams, num_objs), bool)
for i, name in enumerate(cams):
    entry = next(c for c in sample['cameras'] if c['name']==name)
    for pid in entry['visible_ids']:
        vis[i, all_pids.index(pid)] = True
O_mat = np.zeros((num_cams, 4*num_objs))
dirs = [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]
for i, name in enumerate(cams):
    dvec = axes[name]
    for j in range(num_objs):
        if not vis[i,j]: continue
        base = 4*j
        for k, ud in enumerate(dirs):
            O_mat[i, base+k] = 0.5 * (1.0 + np.dot(dvec, ud))

# Growth rates g_k
np.random.seed(123)
g = np.random.uniform(0.5, 2.0, size=4*num_objs)

# ========== Compute optimal λ ==========
lam_opt = hybrid_opt(compute_aoi_objective, mu, O_mat, g, top_k=10)

# ========== Simulate and track Face 0 ==========
def simulate_track_face0(lam, mu, O_mat, g, T=200.0, seed=42):
    random.seed(seed); np.random.seed(seed)
    num_cams = len(lam); num_feat = len(g)
    events = []; evt_id = 0
    for i in range(num_cams):
        t = 0.0
        while True:
            dt = random.expovariate(lam[i])
            t += dt
            if t > T: break
            heapq.heappush(events, (t, evt_id, 'arrival', {'cam':i, 'arrival_time':t}))
            evt_id += 1

    processing = None; waiting = None
    aoi = np.zeros(num_feat)
    last_t = 0.0
    times = [0.0]; aoi0 = [0.0]

    while events:
        t, _, etype, info = heapq.heappop(events)
        if t > T:
            t = T
        # advance AoI
        dt = t - last_t
        aoi += g * dt
        last_t = t
        times.append(t); aoi0.append(aoi[0])

        if etype == 'arrival':
            cam = info['cam']; atime = info['arrival_time']
            if processing is None:
                processing = {'cam':cam,'arrival_time':atime}
                ptime = random.expovariate(mu[cam])
                heapq.heappush(events, (t+ptime, evt_id, 'completion', processing.copy()))
                evt_id += 1
            else:
                waiting = {'cam':cam,'arrival_time':atime}

        else:  # completion
            cam = info['cam']; atime = info['arrival_time']
            delay = t - atime
            # reset for face 0 only
            if random.random() <= O_mat[cam, 0]:
                aoi[0] = g[0] * delay
            times.append(t); aoi0.append(aoi[0])

            if waiting:
                nxt = waiting; waiting = None
                processing = nxt
                ptime = random.expovariate(mu[nxt['cam']])
                heapq.heappush(events, (t+ptime, evt_id, 'completion', processing.copy()))
                evt_id += 1
            else:
                processing = None

        if last_t >= T:
            break

    return times, aoi0

times, aoi0_vals = simulate_track_face0(lam_opt, mu, O_mat, g, T=200.0, seed=42)

# ========== Plot ==========
plt.figure()
plt.plot(times, aoi0_vals)
plt.xlabel('Time')
plt.ylabel('AoI (Face 0)')
plt.title('AoI Evolution for Face 0')
plt.tight_layout()
plt.show()
