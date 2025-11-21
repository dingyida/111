import os
import glob
import json
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ================= USER SETTINGS =================
data_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\output"
Lambda_max = 10.0
EPS = 1e-6
penalty_coef = 1e3  # linear penalty for PSO constraint
swarm_size = 30
pso_max_iter = 10
slsqp_max_iter = 50
top_k_for_slsqp = 10
n_random_samples = 30  # for random search and multi-start SLSQP


# ================== HELPER FUNCTIONS ==================

def walker_idx_to_pid(idx: int) -> int:
    return 24 + 2 * idx


def compute_average_speeds(folder):
    tracks = {}
    files = sorted(glob.glob(os.path.join(folder, 'frame_*.json')))
    for js in files:
        data = json.load(open(js))
        for cam in data['cameras']:
            if cam['name'] == 'fixed_camera': continue
            pid = walker_idx_to_pid(int(cam['name'].split('_')[1]))
            xyz = cam['location']
            pos = np.array([xyz['x'], xyz['y'], xyz['z']])
            tracks.setdefault(pid, []).append(pos)
    avg = {}
    for pid, pts in tracks.items():
        if len(pts) < 2:
            avg[pid] = 0.0
        else:
            dist = sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
            avg[pid] = dist / (len(pts) - 1)
    return avg


def euler_to_vector(pitch_deg, yaw_deg):
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    return np.array([
        math.cos(p) * math.cos(y),
        math.cos(p) * math.sin(y),
        math.sin(p)
    ])


def compute_steady_state(lam, mu):
    L = lam.sum()
    A = np.sum(lam / (L + mu))
    B = np.sum(lam / (mu * (L + mu)))
    pi0 = (1 - A) / (1 + L * B)
    pii0 = pi0 * lam / ((L + mu) * (1 - A))
    pii1 = pii0 * (L / mu)
    return pi0, pii0, pii1


def compute_aoi_objective(lam, mu, A_mat, g):
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    phi = mu * (pii0 + pii1) + EPS
    L = lam.sum()
    pit = pii0 + pii1
    D = np.zeros_like(mu)
    for i in range(len(mu)):
        t1 = pi0 * lam[i] / mu[i]
        t2 = sum(
            pit[k] * lam[i] * (mu[k] / (mu[k] + L)) *
            (1 / (mu[k] + L) + 1 / mu[i]) for k in range(len(mu))
        )
        D[i] = (t1 + t2) / phi[i]
    mask = A_mat.sum(0) > 0
    A_vis = A_mat[:, mask]
    g_vis = g[mask]
    Phi = A_vis.T.dot(phi)
    Num = A_vis.T.dot(phi * D)
    AoIvec = (Num / Phi) + 1.0 / Phi
    return float(g_vis.dot(AoIvec))


def gradient_lambda(lam, mu, A_mat, g, eps=1e-8):
    base = compute_aoi_objective(lam, mu, A_mat, g)
    grad = np.zeros_like(lam)
    for i in range(len(lam)):
        step = eps * max(1.0, abs(lam[i]))
        lam2 = lam.copy();
        lam2[i] += step
        grad[i] = (compute_aoi_objective(lam2, mu, A_mat, g) - base) / step
    return grad


# ================== PSO + SLSQP ITER COUNT WRAPPERS ==================

def pso_optimize(obj, mu, A_mat, g):
    N = len(mu)
    X = np.clip(np.random.rand(swarm_size, N), EPS, None)
    V = np.zeros_like(X)
    pbest, pval = X.copy(), np.full(swarm_size, np.inf)

    def penalized(lam):
        loss = obj(lam, mu, A_mat, g)
        cons = max(0.0, lam.sum() - Lambda_max)
        return loss + penalty_coef * cons

    for i in range(swarm_size):
        pval[i] = penalized(X[i])
    for _ in range(pso_max_iter):
        gbest_i = np.argmin(pval)
        for i in range(swarm_size):
            r1, r2 = np.random.rand(N), np.random.rand(N)
            V[i] = 0.5 * V[i] + 1.5 * r1 * (pbest[i] - X[i]) + 1.5 * r2 * (X[gbest_i] - X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], EPS, None)
            v = penalized(X[i])
            if v < pval[i]:
                pval[i], pbest[i] = v, X[i].copy()
    return pbest, pval


def pso_slsqp(mu, A_mat, g):
    start = time.time()
    # Global PSO search
    pbest, pval = pso_optimize(compute_aoi_objective, mu, A_mat, g)
    # Local SLSQP on top K seeds, record best nit
    idxs = np.argsort(pval)[:top_k_for_slsqp]
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons_fcn = {'type': 'eq', 'fun': lambda lam: lam.sum() - Lambda_max}
    best, best_val, best_nit = None, np.inf, 0
    for i in idxs:
        res = minimize(
            compute_aoi_objective, pbest[i],
            args=(mu, A_mat, g),
            method='SLSQP',
            bounds=bounds,
            constraints=[cons_fcn],
            options={'ftol': 1e-6, 'maxiter': slsqp_max_iter}
        )
        if res.success and res.fun < best_val:
            best, best_val = res.x.copy(), res.fun
            best_nit = res.nit
    elapsed = time.time() - start
    total_iters = pso_max_iter + best_nit
    return best, best_val, elapsed, total_iters


def multistart_slsqp(mu, A_mat, g, n_starts=n_random_samples):
    start = time.time()
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons_fcn = {'type': 'eq', 'fun': lambda lam: lam.sum() - Lambda_max}
    best, best_val, best_nit = None, np.inf, 0
    for _ in range(n_starts):
        x0 = np.random.rand(len(mu))
        x0 = x0 / x0.sum() * Lambda_max
        res = minimize(
            compute_aoi_objective, x0,
            args=(mu, A_mat, g),
            method='SLSQP',
            bounds=bounds,
            constraints=[cons_fcn],
            options={'ftol': 1e-6, 'maxiter': slsqp_max_iter}
        )
        if res.success and res.fun < best_val:
            best, best_val = res.x.copy(), res.fun
            best_nit = res.nit
    elapsed = time.time() - start
    return best, best_val, elapsed, best_nit


def random_search(mu, A_mat, g, n_samples=n_random_samples):
    start = time.time()
    best_val = np.inf
    best_lam = None
    for _ in range(n_samples):
        x = np.random.rand(len(mu))
        lam = x / x.sum() * Lambda_max
        val = compute_aoi_objective(lam, mu, A_mat, g)
        if val < best_val:
            best_val, best_lam = val, lam.copy()
    elapsed = time.time() - start
    return best_lam, best_val, elapsed, n_samples


# ================== MAIN BENCHMARK & PLOTTING ==================

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(data_folder, 'frame_*.json')))
    if not files:
        raise RuntimeError("No frame_*.json files found in data_folder.")

    avg_speeds = compute_average_speeds(data_folder)
    sample = json.load(open(files[0]))
    all_pids = sorted({pid for cam in sample['cameras'] for pid in cam.get('visible_ids', [])})
    avg_mean = np.mean(list(avg_speeds.values())) if avg_speeds else 0.0
    speeds_const = np.array([avg_speeds.get(pid, avg_mean) for pid in all_pids])
    g_const = np.repeat(speeds_const, 4)

    cams0 = sorted(c['name'] for c in sample['cameras'])
    num_cams = len(cams0)
    mu = np.random.uniform(0.5, 2.0, size=num_cams)

    obs_dirs = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    results = []

    # Iterate all frames
    for fp in files:
        frame = int(os.path.basename(fp).split('_')[1].split('.')[0])
        data = json.load(open(fp))
        cams = sorted(c['name'] for c in data['cameras'])

        # Build A_mat
        axes = {c['name']: euler_to_vector(c['rotation']['pitch'], c['rotation']['yaw'])
                for c in data['cameras']}
        vis = np.zeros((len(cams), len(all_pids)), dtype=bool)
        for i, name in enumerate(cams):
            entry = next(c for c in data['cameras'] if c['name'] == name)
            for pid in entry.get('visible_ids', []):
                j = all_pids.index(pid)
                vis[i, j] = True

        A_mat = np.zeros((len(cams), 4 * len(all_pids)))
        for i, name in enumerate(cams):
            dvec = axes[name]
            for j in range(len(all_pids)):
                if not vis[i, j]: continue
                base = 4 * j
                for k, ud in enumerate(obs_dirs):
                    A_mat[i, base + k] = 0.5 * (1.0 + dvec.dot(ud))

        # Run benchmarks
        lam_rnd, val_rnd, t_rnd, iter_rnd = random_search(mu, A_mat, g_const)
        lam_pso, val_pso, t_pso, iter_pso = pso_slsqp(mu, A_mat, g_const)
        lam_ms, val_ms, t_ms, iter_ms = multistart_slsqp(mu, A_mat, g_const)

        results.append({
            'frame': frame,
            'random_val': val_rnd, 'random_time': t_rnd, 'random_iter': iter_rnd,
            'pso_val': val_pso, 'pso_time': t_pso, 'pso_iter': iter_pso,
            'multi_val': val_ms, 'multi_time': t_ms, 'multi_iter': iter_ms
        })

        print(f"Frame {frame}: "
              f"Random={val_rnd:.4f} ({t_rnd:.2f}s,{iter_rnd} it), "
              f"PSO+SLSQP={val_pso:.4f} ({t_pso:.2f}s,{iter_pso} it), "
              f"Multi-SLSQP={val_ms:.4f} ({t_ms:.2f}s,{iter_ms} it)")

    # Save and load into DataFrame
    df = pd.DataFrame(results).sort_values('frame')
    df.to_csv('benchmark_results.csv', index=False)
    print("Saved detailed results to benchmark_results.csv")

    # Plot AoI objective vs Frame
    plt.figure()
    plt.plot(df['frame'], df['random_val'], label='Random-only')
    plt.plot(df['frame'], df['pso_val'], label='PSO+SLSQP')
    plt.plot(df['frame'], df['multi_val'], label='Multi-start SLSQP')
    plt.xlabel('Frame');
    plt.ylabel('AoI Objective');
    plt.legend()
    plt.tight_layout();
    plt.show()

    # Plot time cost vs Frame
    plt.figure()
    plt.plot(df['frame'], df['random_time'], label='Random-only')
    plt.plot(df['frame'], df['pso_time'], label='PSO+SLSQP')
    plt.plot(df['frame'], df['multi_time'], label='Multi-start SLSQP')
    plt.xlabel('Frame');
    plt.ylabel('Time (s)');
    plt.legend()
    plt.tight_layout();
    plt.show()

    # Plot iteration count vs Frame
    plt.figure()
    plt.plot(df['frame'], df['random_iter'], label='Random-iters')
    plt.plot(df['frame'], df['pso_iter'], label='PSO+SLSQP-iters')
    plt.plot(df['frame'], df['multi_iter'], label='Multi-start SLSQP-iters')
    plt.xlabel('Frame');
    plt.ylabel('Iterations');
    plt.legend()
    plt.tight_layout();
    plt.show()
