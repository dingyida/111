import os
import glob
import json
import math
import heapq
import numpy as np
from scipy.optimize import minimize

# ================= USER SETTINGS =================
data_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\7walker"
Lambda_max               = 20
EPS                      = 1e-6
swarm_size               = 30
pso_max_iter             = 100
slsqp_max_iter           = 100
top_k_for_slsqp          = 10
T_end                    = 200.0   # total simulation time

# ================== MODEL HELPERS ==================
def compute_steady_state(lam, mu):
    L = lam.sum()
    A = np.sum(lam / (L + mu))
    B = np.sum(lam / (mu * (L + mu)))
    pi0   = (1 - A) / (1 + L * B)
    pii0  = pi0 * lam / ((L + mu) * (1 - A))
    pii1  = pii0 * (L / mu)
    return pi0, pii0, pii1

def pso_optimize(obj, lam_init, mu, O_mat, g,
                 swarm_size, max_iter, w=0.5, c1=1.5, c2=1.5):
    N = len(mu)
    X = np.clip(np.random.rand(swarm_size, N) * lam_init, EPS, None)
    V = np.zeros_like(X)
    pbest = X.copy()
    def penalized(x):
        return obj(x, mu, O_mat, g) + 1e3 * max(0.0, x.sum() - Lambda_max)
    pval = np.array([penalized(x) for x in X])
    for _ in range(max_iter):
        gbest = pbest[np.argmin(pval)]
        for i in range(swarm_size):
            r1 = np.random.rand(N)
            r2 = np.random.rand(N)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], EPS, None)
            val = penalized(X[i])
            if val < pval[i]:
                pval[i] = val
                pbest[i] = X[i].copy()
    return pbest, pval

def hybrid_opt(obj, mu, O_mat, g, top_k):
    lam_init = np.ones_like(mu) * (Lambda_max / len(mu))
    pbest, pval = pso_optimize(obj, lam_init, mu, O_mat, g,
                               swarm_size, pso_max_iter)
    idxs = np.argsort(pval)[:top_k]
    best, val = None, np.inf
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons = {'type':'eq', 'fun': lambda x: x.sum() - Lambda_max}
    for i in idxs:
        res = minimize(lambda x: obj(x, mu, O_mat, g),
                       pbest[i], method='SLSQP', bounds=bounds,
                       constraints=[cons],
                       options={'ftol':1e-6, 'maxiter':slsqp_max_iter})
        if res.success and res.fun < val:
            best, val = res.x.copy(), res.fun
    return best

def compute_aoi_objective(lam, mu, O_mat, g):
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    phi = mu * (pii0 + pii1) + EPS
    mask  = O_mat.sum(axis=0) > 0
    O_vis = O_mat[:, mask]
    g_vis = g[mask]
    Phi = O_vis.T.dot(phi)
    L = lam.sum()
    busy_frac = pii0 + pii1
    vis_bin = (O_vis > 0).astype(float)
    P = vis_bin.T.dot(busy_frac)
    AoIvec = (1.0 / L) + (1.0 + 2.0 * P) / (Phi + EPS)
    return float(g_vis.dot(AoIvec))


# ================== MAIN SCRIPT ==================
if __name__ == "__main__":
    # 1) Load first frame to build O_mat and O_bin
    files = sorted(glob.glob(os.path.join(data_folder, 'frame_*.json')))
    sample = json.load(open(files[0]))
    cams = sorted(cam['name'] for cam in sample['cameras'])
    num_cams = len(cams)
    all_pids = sorted({pid for cam in sample['cameras'] for pid in cam.get('visible_ids', [])})
    num_objs = len(all_pids)

    # Build O_mat for optimization (4‐dir per object)
    obs_dirs = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]])
    O_mat = np.zeros((num_cams, 4 * num_objs))
    for i, cam in enumerate(sample['cameras']):
        p = math.radians(cam['rotation']['pitch'])
        y = math.radians(cam['rotation']['yaw'])
        dvec = np.array([math.cos(p)*math.cos(y), math.cos(p)*math.sin(y), math.sin(p)])
        for pid in cam.get('visible_ids', []):
            j = all_pids.index(pid)
            base = 4*j
            for k, ud in enumerate(obs_dirs):
                O_mat[i, base+k] = 0.5 * (1.0 + dvec.dot(ud))

    # Build O_bin for simulation
    O_bin = np.zeros((num_cams, num_objs))
    for i, cam in enumerate(sample['cameras']):
        for pid in cam.get('visible_ids', []):
            j = all_pids.index(pid)
            O_bin[i, j] = 1.0

    # 2) Sample fixed μ ∈ Uniform[0.5,2]
    np.random.seed(42)
    mu = np.random.uniform(0.5, 2.0, size=num_cams)

    # 3) Simplify g_const for optimization features
    np.random.seed(123)
    g_const = np.random.uniform(0.5, 2.0, size=4 * num_objs)

    # 4) Compute optimal λ via PSO+SLSQP
    lam_opt = hybrid_opt(compute_aoi_objective, mu, O_mat, g_const, top_k_for_slsqp)

    # 5) Simulation‐time object weights g_j ∼ Uniform[0.5,2]
    np.random.seed(999)
    g_objs = np.random.uniform(0.5, 2.0, size=num_objs)

    # Prepare event queue
    ARRIVAL, COMPLETION = 0, 1
    events = []
    for i in range(num_cams):
        t = 0.0
        while True:
            t += np.random.exponential(1.0 / lam_opt[i])
            if t > T_end:
                break
            heapq.heappush(events, (t, ARRIVAL, i))

    # Simulation state
    processing = None   # (cam_i, arrival_time)
    waiting = None      # (cam_i, arrival_time)
    AoI = np.zeros(num_objs)
    area = np.zeros(num_objs)
    last_t = 0.0

    # Event‐driven simulation (standard AoI)
    while events:
        t, etype, data = heapq.heappop(events)
        if t > T_end:
            break

        # Advance time
        dt = t - last_t
        area += AoI * dt
        AoI += dt
        last_t = t

        if etype == ARRIVAL:
            cam = data
            arr_time = t
            if processing is None:
                processing = (cam, arr_time)
                proc = np.random.exponential(1.0 / mu[cam])
                heapq.heappush(events, (t + proc, COMPLETION, (cam, arr_time)))
            else:
                waiting = (cam, arr_time)
        else:  # COMPLETION
            cam_p, arr_p = data
            # Reset AoI_j with probability O_bin[cam_p, j]
            for j in range(num_objs):
                if np.random.rand() < O_bin[cam_p, j]:
                    AoI[j] = t - arr_p

            # Move waiting to processing
            if waiting is not None:
                cam_w, arr_w = waiting
                processing = (cam_w, arr_w)
                waiting = None
                proc = np.random.exponential(1.0 / mu[cam_w])
                heapq.heappush(events, (t + proc, COMPLETION, (cam_w, arr_w)))
            else:
                processing = None

    # Finish up to T_end
    if last_t < T_end:
        dt = T_end - last_t
        area += AoI * dt
        AoI += dt

    # Compute average AoI per object
    avg_AoI_per_obj = area / T_end

    # Weighted empirical mean AoI
    empirical_weighted_mean = (g_objs * avg_AoI_per_obj).sum() / g_objs.sum()

    # Analytical AoI_j (bare)
    pi0, pii0, pii1 = compute_steady_state(lam_opt, mu)
    phi = mu * (pii0 + pii1) + EPS
    busy_frac = pii0 + pii1
    P = O_bin.T.dot(busy_frac)
    Phi = O_bin.T.dot(phi)
    aois_bare = (1.0 / lam_opt.sum()) + (1.0 + 2.0 * P) / (Phi + EPS)

    # Weighted analytical mean AoI
    analytical_weighted_mean = (g_objs * aois_bare).sum() / g_objs.sum()

    print(f"Empirical weighted‐mean AoI (sim): {empirical_weighted_mean:.4f}")
    print(f"Analytical weighted‐mean AoI:    {analytical_weighted_mean:.4f}")
