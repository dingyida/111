import os
import glob
import json
import math
import numpy as np
from scipy.optimize import minimize

# ================= USER SETTINGS =================
data_folder              = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\7walker"
Lambda_max               = 10
EPS                      = 1e-6
penalty_coef             = 1e3      # coefficient for PSO penalty term
large_change_threshold   = 0.15     # threshold for full re-opt trigger
recompute_hessian_every  = 20       # max incremental steps before Hessian rebuild
swarm_size               = 30
pso_max_iter             = 100
slsqp_max_iter           = 50
top_k_for_slsqp          = 10

# ================== BASE HELPERS ==================

def walker_idx_to_pid(idx: int) -> int:
    """Map walker index (0-based in filenames) to persistent PID."""
    return 24 + 2 * idx

def compute_average_speeds(folder):
    """Compute per-PID average speed over all frames."""
    tracks = {}
    files = sorted(glob.glob(os.path.join(folder, 'frame_*.json')))
    for js in files:
        frame = int(os.path.basename(js).split('_')[1].split('.')[0])
        data  = json.load(open(js))
        for cam in data['cameras']:
            if cam['name'] == 'fixed_camera':
                continue
            pid = walker_idx_to_pid(int(cam['name'].split('_')[1]))
            pos = np.array([cam['location']['x'],
                            cam['location']['y'],
                            cam['location']['z']])
            tracks.setdefault(pid, []).append((frame, pos))
    avg = {}
    for pid, pts in tracks.items():
        pts = sorted(pts)
        if len(pts) < 2:
            avg[pid] = 0.0
            continue
        dist, dt = 0.0, 0
        for (f0,p0),(f1,p1) in zip(pts, pts[1:]):
            dist += np.linalg.norm(p1 - p0)
            dt   += f1 - f0
        avg[pid] = dist / dt if dt > 0 else 0.0
    return avg

def euler_to_vector(pitch_deg, yaw_deg):
    """Convert Euler angles (degrees) to 3D unit vector."""
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    return np.array([math.cos(p)*math.cos(y),
                     math.cos(p)*math.sin(y),
                     math.sin(p)])

# ================== AOI MODEL ==================

def compute_steady_state(lam, mu):
    L   = lam.sum()
    A   = np.sum(lam / (L + mu))
    B   = np.sum(lam / (mu * (L + mu)))
    pi0 = (1 - A) / (1 + L * B)
    pii0 = pi0 * lam / ((L + mu) * (1 - A))
    pii1 = pii0 * (L / mu)
    return pi0, pii0, pii1

def compute_aoi_objective(lam, mu, A_mat, g):
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    phi = mu * (pii0 + pii1) + EPS
    L   = lam.sum()
    pit = pii0 + pii1

    # directional delay contributions
    D = np.zeros_like(mu)
    for i in range(len(mu)):
        t1 = pi0 * lam[i] / mu[i]
        t2 = sum(
            pit[k] * lam[i] * (mu[k] / (mu[k] + L)) *
            (1/(mu[k] + L) + 1/mu[i])
            for k in range(len(mu))
        )
        D[i] = (t1 + t2) / phi[i]

    mask   = A_mat.sum(0) > 0
    A_vis  = A_mat[:, mask]
    g_vis  = g[mask]

    Phi    = A_vis.T.dot(phi)
    Num    = A_vis.T.dot(phi * D)
    AoIvec = (Num / Phi) + 1.0/Phi
    return float(g_vis.dot(AoIvec))

def gradient_lambda(lam, mu, A_mat, g, eps=1e-8):
    """Finite-difference gradient ∂(AoI)/∂λᵢ."""
    base = compute_aoi_objective(lam, mu, A_mat, g)
    grad = np.zeros_like(lam)
    for i in range(len(lam)):
        step = eps * max(1.0, abs(lam[i]))
        lam2 = lam.copy()
        lam2[i] += step
        grad[i] = (compute_aoi_objective(lam2, mu, A_mat, g) - base) / step
    return grad

# ================== PSO + HYBRID GLOBAL OPT ==================

def pso_optimize(obj, mu, A_mat, g,
                 swarm_size=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    N = len(mu)
    X = np.clip(np.random.rand(swarm_size, N), EPS, None)
    V = np.zeros_like(X)
    pbest = X.copy()
    pval  = np.full(swarm_size, np.inf)

    def penalized(lam):
        loss = obj(lam, mu, A_mat, g)
        cons = max(0.0, lam.sum() - Lambda_max)
        return loss + penalty_coef * cons

    for i in range(swarm_size):
        pval[i] = penalized(X[i])

    for _ in range(max_iter):
        gbest_i = np.argmin(pval)
        for i in range(swarm_size):
            r1, r2 = np.random.rand(N), np.random.rand(N)
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(X[gbest_i]-X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], EPS, None)
            v = penalized(X[i])
            if v < pval[i]:
                pval[i], pbest[i] = v, X[i].copy()
    return pbest, pval

def hybrid_opt(obj, mu, A_mat, g, top_k=10):
    pbest, pval = pso_optimize(obj, mu, A_mat, g,
                               swarm_size=swarm_size,
                               max_iter=pso_max_iter)
    idxs   = np.argsort(pval)[:top_k]
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons   = {'type': 'eq', 'fun': lambda lam: lam.sum() - Lambda_max}
    best, val = None, np.inf
    for i in idxs:
        res = minimize(obj, pbest[i], args=(mu, A_mat, g),
                       method='SLSQP', bounds=bounds,
                       constraints=[cons],
                       options={'ftol':1e-6, 'maxiter':slsqp_max_iter})
        if res.fun < val:
            best, val = res.x.copy(), res.fun
    return best

# ============ INCREMENTAL (KKT + IFT) ============

class IncrementalOptimizer:
    def __init__(self, mu, Lambda_max):
        self.mu = mu
        self.Lambda_max = Lambda_max
        self.lam = None  # primal variables
        self.nu = 0.0  # multiplier for ∑λ = Λ_max
        self.alpha = None  # multipliers for λ_i ≥ EPS
        self.H = None  # Hessian ∇²f
        self.count = 0
        self.prev_A = None

    def build_hessian(self, lam, A_mat, g):
        N = len(lam)
        H = np.zeros((N, N))
        grad0 = gradient_lambda(lam, self.mu, A_mat, g)
        for i in range(N):
            lam2 = lam.copy()
            lam2[i] += EPS
            grad2 = gradient_lambda(lam2, self.mu, A_mat, g)
            H[:, i] = (grad2 - grad0) / EPS
        return 0.5 * (H + H.T)

    def initialize(self, A_mat, g):
        # 1) Global PSO+SLSQP solve (with bounds λ≥EPS and ∑λ=Λ_max)
        self.lam = hybrid_opt(compute_aoi_objective,
                              self.mu, A_mat, g,
                              top_k=top_k_for_slsqp)
        # ensure strict feasibility
        self.lam = np.maximum(self.lam, EPS)
        # enforce sum constraint exactly
        self.lam *= self.Lambda_max / (self.lam.sum() + EPS)

        # 2) Compute ∇f and Hessian
        grad0 = gradient_lambda(self.lam, self.mu, A_mat, g)
        self.H = self.build_hessian(self.lam, A_mat, g)

        # 3) Solve nu from stationarity: ∇f = ν·1 + α
        #    First assume α=0 for λ_i>ε, so:
        self.nu = float(np.dot(grad0, np.ones_like(grad0)) / len(grad0))

        # 4) Compute α_i = max( ∂f/∂λ_i - ν, 0 ), enforce complementarity
        alpha = grad0 - self.nu
        self.alpha = np.maximum(alpha, 0.0)
        self.alpha[self.lam > EPS + 1e-8] = 0.0

        self.count = 0
        self.prev_A = A_mat.copy()

    def update(self, A_mat, g):
        if self.lam is None:
            self.initialize(A_mat, g)
            return self.lam

        # Rebuild if visibility changed drastically or too many steps
        relA = np.linalg.norm(A_mat - self.prev_A) / (np.linalg.norm(self.prev_A) + EPS)
        if relA > large_change_threshold or self.count >= recompute_hessian_every:
            self.initialize(A_mat, g)
            return self.lam

        # Compute change in gradient due to A
        grad_old = gradient_lambda(self.lam, self.mu, self.prev_A, g)
        grad_new = gradient_lambda(self.lam, self.mu, A_mat, g)
        g_lO = grad_new - grad_old

        N = len(self.lam)
        # Build full KKT matrix:
        #   [ H    -1    -I  ]
        #   [1^T    0     0  ]
        #   [Diag(α) 0 Diag(λ - ε)]
        K_top = np.hstack([self.H, -np.ones((N, 1)), -np.eye(N)])
        K_mid = np.hstack([np.ones((1, N)), np.zeros((1, 1)), np.zeros((1, N))])
        K_bot = np.hstack([np.diag(self.alpha), np.zeros((N, 1)), np.diag(self.lam - EPS)])
        K_full = np.vstack([K_top, K_mid, K_bot])

        # Right-hand side:
        rhs = np.concatenate([-g_lO,
                              [0.0],
                              np.zeros(N)])

        # Solve for [dλ; dν; dα]
        try:
            d = np.linalg.solve(K_full, rhs)
        except np.linalg.LinAlgError:
            # fallback to full rebuild
            self.initialize(A_mat, g)
            return self.lam

        dlam = d[:N]
        dnu = d[N:N + 1][0]
        dalpha = d[N + 1:]

        # 1) Update primal & multipliers
        self.lam = self.lam + dlam
        self.nu = self.nu + dnu
        self.alpha = self.alpha + dalpha

        # 2) Enforce λᵢ ≥ ε (via explicit clipping only)
        self.lam = np.maximum(self.lam, EPS)

        # 3) Re-enforce complementarity: αᵢ = 0 for λᵢ > ε
        self.alpha[self.lam > EPS + 1e-8] = 0.0

        self.count += 1
        self.prev_A = A_mat.copy()

        return self.lam

# ================== MAIN LOOP ==================

if __name__ == "__main__":
    # Precompute speeds & prepare frames
    avg_speeds = compute_average_speeds(data_folder)
    avg_mean   = np.mean(list(avg_speeds.values())) if avg_speeds else 0.0

    files  = sorted(glob.glob(os.path.join(data_folder, 'frame_*.json')))
    frames = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]

    sample   = json.load(open(files[0]))
    cams     = sorted(c['name'] for c in sample['cameras'])
    num_cams = len(cams)

    obs_dirs = np.array([[ 1, 0, 0],
                         [ 0, 1, 0],
                         [-1, 0, 0],
                         [ 0,-1, 0]])

    all_pids = []
    for fp in files:
        d = json.load(open(fp))
        for c in d['cameras']:
            all_pids.extend(c.get('visible_ids', []))
    all_pids = sorted(set(all_pids))

    speeds_const = np.array([avg_speeds.get(pid, avg_mean) for pid in all_pids])
    g_const      = np.repeat(speeds_const, 4)

    mu      = np.random.uniform(1.0, 1.0, size=num_cams)
    inc_opt = IncrementalOptimizer(mu, Lambda_max)

    AoI_inc, AoI_prop, AoI_full, AoI_inv = [], [], [], []

    for fp, t in zip(files, frames):
        data = json.load(open(fp))

        # build camera axes
        axes = {
            c['name']: euler_to_vector(c['rotation']['pitch'],
                                       c['rotation']['yaw'])
            for c in data['cameras']
        }

        # boolean visibility mask
        vis = np.zeros((num_cams, len(all_pids)), dtype=bool)
        for i,name in enumerate(cams):
            entry = next(c for c in data['cameras'] if c['name']==name)
            for pid in entry['visible_ids']:
                j = all_pids.index(pid)
                vis[i,j] = True

        # build directional A_mat
        A_mat = np.zeros((num_cams, 4 * len(all_pids)))
        for i,name in enumerate(cams):
            dvec = axes[name]
            for j in range(len(all_pids)):
                if not vis[i,j]: continue
                base = 4*j
                for k, ud in enumerate(obs_dirs):
                    A_mat[i, base+k] = 0.5 * (1.0 + dvec.dot(ud))

        # 1) Incremental KKT/IFT
        lam_inc = inc_opt.update(A_mat, g_const)
        AoI_inc.append(compute_aoi_objective(lam_inc, mu, A_mat, g_const))

        # 2) Proportional allocation
        w       = A_mat.dot(g_const)
        total_w = w.sum()
        if total_w <= EPS:
            lam_prop = np.ones(num_cams) * (Lambda_max/num_cams)
        else:
            lam_prop = Lambda_max * (w / total_w)
        AoI_prop.append(compute_aoi_objective(lam_prop, mu, A_mat, g_const))

        # 3) Full PSO+SLSQP
        lam_full = hybrid_opt(compute_aoi_objective, mu, A_mat, g_const,
                              top_k=top_k_for_slsqp)
        AoI_full.append(compute_aoi_objective(lam_full, mu, A_mat, g_const))

        # 4) Reciprocal-objective (λ ∝ √w)
        sqrt_w = np.sqrt(np.maximum(w, 0.0))
        S      = sqrt_w.sum()
        if S <= EPS:
            lam_inv = np.ones(num_cams) * (Lambda_max/num_cams)
        else:
            lam_inv = Lambda_max * (sqrt_w / S)
        AoI_inv.append(compute_aoi_objective(lam_inv, mu, A_mat, g_const))

        print(f"Frame {t}: "
              f"inc={AoI_inc[-1]:.4f}, "
              f"prop={AoI_prop[-1]:.4f}, "
              f"full={AoI_full[-1]:.4f}, "
              f"inv={AoI_inv[-1]:.4f}")

    # Plot comparison
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(frames, AoI_inc,  label='Incremental')
        plt.plot(frames, AoI_prop, label='Proportional')
        plt.plot(frames, AoI_full, label='PSO+SLSQP')
        plt.plot(frames, AoI_inv,  label='Reciprocal-obj')
        plt.xlabel('Frame')
        plt.ylabel('AoI')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print('Skipping plot: matplotlib unavailable')
