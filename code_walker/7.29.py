import os
import glob
import json
import math
import numpy as np
from scipy.optimize import minimize

# ================= USER SETTINGS =================
data_folder              = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\7walker"
Lambda_max               = 20
EPS                      = 1e-6
penalty_coef             = 1e3      # coefficient for PSO penalty term
large_change_threshold   = 0.15     # threshold for full re-opt trigger
recompute_hessian_every  = 20       # max incremental steps before Hessian rebuild
swarm_size               = 30
pso_max_iter             = 100
slsqp_max_iter           = 100
top_k_for_slsqp          = 10

# ================== BASE HELPERS ==================
def walker_idx_to_pid(idx: int) -> int:
    return 24 + 2 * idx



def euler_to_vector(pitch_deg, yaw_deg):
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    return np.array([math.cos(p)*math.cos(y),
                     math.cos(p)*math.sin(y),
                     math.sin(p)])

# ================== AOI MODEL ==================
def compute_steady_state(lam, mu):
    L = lam.sum()
    A = np.sum(lam / (L + mu))
    B = np.sum(lam / (mu * (L + mu)))

    # π₀: probability server is idle
    pi0 = (1 - A) / (1 + L * B)

    # πᵢ⁰: fraction of time server is serving from camera i AND waiting slot is EMPTY
    pii0 = pi0 * lam / ((L + mu) * (1 - A))

    # πᵢ¹: fraction of time server is serving from camera i AND waiting slot has ≥1 frame
    pii1 = pii0 * (L / mu)

    return pi0, pii0, pii1

def compute_aoi_objective(lam, mu, O_mat, g):
    # 1) Steady‐state → π₀, πᵢ⁰, πᵢ¹
    pi0, pii0, pii1 = compute_steady_state(lam, mu)

    # 2) Effective per‐camera completion rates φᵢ
    phi = mu * (pii0 + pii1) + EPS

    # 3) Keep only objects actually seen
    mask  = O_mat.sum(axis=0) > 0
    O_vis = O_mat[:, mask]
    g_vis = g[mask]

    # 4) Per‐object aggregate update rate Φ_j = Σ_i O_ij * φ_i
    Phi = O_vis.T.dot(phi)                         # shape: (n_objs,)

    # 5) Total arrival rate Λ = Σ_i λ_i
    L = lam.sum()

    # 6) P_j = Σ_{i:O_ij>0} (πᵢ⁰ + πᵢ¹)
    busy_frac = (pii0 + pii1)                      # shape: (n_cams,)
    vis_bin   = (O_vis > 0).astype(float)          # (n_cams, n_objs)
    P         = vis_bin.T.dot(busy_frac)           # shape: (n_objs,)

    # 7) New AoI: 1/Λ + (1 + 2P_j)/Φ_j
    AoIvec = (1.0 / L) + (1.0 + 2.0 * P) / (Phi + EPS)

    # 8) Weighted sum over objects
    return float(g_vis.dot(AoIvec))

def gradient_lambda(lam, mu, O_mat, g, eps=1e-8):
    base = compute_aoi_objective(lam, mu, O_mat, g)
    grad = np.zeros_like(lam)
    for i in range(len(lam)):
        step = eps * max(1.0, abs(lam[i]))
        lam2 = lam.copy(); lam2[i] += step
        grad[i] = (compute_aoi_objective(lam2, mu, O_mat, g) - base) / step
    return grad

# ================= PSO + HYBRID GLOBAL OPTIM =================
def pso_optimize(obj, mu, O_mat, g,
                 swarm_size=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    N = len(mu)
    X = np.clip(np.random.rand(swarm_size, N), EPS, None)
    V = np.zeros_like(X)
    pbest = X.copy()
    pval  = np.full(swarm_size, np.inf)

    def penalized(lam):
        loss = obj(lam, mu, O_mat, g)
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

def hybrid_opt(obj, mu, O_mat, g, top_k=10):
    pbest, pval = pso_optimize(obj, mu, O_mat, g,
                               swarm_size=swarm_size,
                               max_iter=pso_max_iter)
    idxs   = np.argsort(pval)[:top_k]
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons   = {'type': 'eq', 'fun': lambda lam: lam.sum() - Lambda_max}
    best, val = None, np.inf
    for i in idxs:
        res = minimize(obj, pbest[i], args=(mu, O_mat, g),
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
        self.lam = None
        self.nu = 0.0
        self.alpha = None
        self.H = None
        self.count = 0
        self.prev_O = None

    def build_hessian(self, lam, O_mat, g):
        N = len(lam)
        H = np.zeros((N, N))
        grad0 = gradient_lambda(lam, self.mu, O_mat, g)
        for i in range(N):
            lam2 = lam.copy(); lam2[i] += EPS
            grad2 = gradient_lambda(lam2, self.mu, O_mat, g)
            H[:, i] = (grad2 - grad0) / EPS
        return 0.5 * (H + H.T)

    def initialize(self, O_mat, g):
        self.lam = hybrid_opt(compute_aoi_objective,
                              self.mu, O_mat, g,
                              top_k=top_k_for_slsqp)
        self.lam = np.maximum(self.lam, EPS)
        self.lam *= self.Lambda_max / (self.lam.sum() + EPS)

        grad0 = gradient_lambda(self.lam, self.mu, O_mat, g)
        self.H = self.build_hessian(self.lam, O_mat, g)

        self.nu = float(np.dot(grad0, np.ones_like(grad0)) / len(grad0))
        alpha = grad0 - self.nu
        self.alpha = np.maximum(alpha, 0.0)
        self.alpha[self.lam > EPS + 1e-8] = 0.0

        self.count = 0
        self.prev_O = O_mat.copy()

    def update(self, O_mat, g):
        if self.lam is None:
            self.initialize(O_mat, g)
            return self.lam

        relO = np.linalg.norm(O_mat - self.prev_O) / (
               np.linalg.norm(self.prev_O) + EPS)
        if relO > large_change_threshold or self.count >= recompute_hessian_every:
            self.initialize(O_mat, g)
            return self.lam

        grad_old = gradient_lambda(self.lam, self.mu, self.prev_O, g)
        grad_new = gradient_lambda(self.lam, self.mu, O_mat, g)
        g_lO = grad_new - grad_old

        N = len(self.lam)
        K_top = np.hstack([self.H, -np.ones((N, 1)), -np.eye(N)])
        K_mid = np.hstack([np.ones((1, N)), np.zeros((1, 1)), np.zeros((1, N))])
        K_bot = np.hstack([np.diag(self.alpha),
                           np.zeros((N, 1)),
                           np.diag(self.lam - EPS)])
        K_full = np.vstack([K_top, K_mid, K_bot])

        rhs = np.concatenate([
            -g_lO,
            [0.0],
            np.zeros(N)
        ])

        try:
            d = np.linalg.solve(K_full, rhs)
        except np.linalg.LinAlgError:
            self.initialize(O_mat, g)
            return self.lam

        dlam   = d[:N]
        dnu    = d[N]
        dalpha = d[N+1:]

        self.lam    = np.maximum(self.lam + dlam, EPS)
        self.nu    += dnu
        self.alpha  = np.maximum(self.alpha + dalpha, 0.0)
        self.alpha[self.lam > EPS + 1e-8] = 0.0

        self.count += 1
        self.prev_O = O_mat.copy()
        return self.lam




# ================== MAIN LOOP ==================
if __name__ == "__main__":
    # List frame files
    files  = sorted(glob.glob(os.path.join(data_folder, 'frame_*.json')))
    frames = [int(os.path.basename(f).split('_')[1].split('.')[0])
              for f in files]

    # Camera names
    sample   = json.load(open(files[0]))
    cams     = sorted(c['name'] for c in sample['cameras'])
    num_cams = len(cams)

    # Fixed μ vector
    np.random.seed(42)
    mu = np.random.uniform(0.5, 2.0, size=num_cams)
    inc_opt = IncrementalOptimizer(mu, Lambda_max)

    # Build list of object IDs
    all_pids = sorted({ pid
        for fp in files
        for c in json.load(open(fp))['cameras']
        for pid in c.get('visible_ids', []) })

    # Simplify g_const: random in [0.5,2] for each feature
    np.random.seed(123)
    g_const = np.random.uniform(0.5, 2.0, size=4*len(all_pids))

    # Collect AoI metrics
    AoI_inc, AoI_prop, AoI_full  = [], [], []

    for fp, t in zip(files, frames):
        data = json.load(open(fp))
        axes = {c['name']:
                euler_to_vector(c['rotation']['pitch'],
                                c['rotation']['yaw'])
                for c in data['cameras']}

        vis = np.zeros((num_cams, len(all_pids)), dtype=bool)
        for i,name in enumerate(cams):
            entry = next(c for c in data['cameras'] if c['name']==name)
            for pid in entry['visible_ids']:
                j = all_pids.index(pid)
                vis[i,j] = True

        O_mat = np.zeros((num_cams, 4*len(all_pids)))
        for i,name in enumerate(cams):
            dvec = axes[name]
            for j in range(len(all_pids)):
                if not vis[i,j]: continue
                base = 4*j
                for k, ud in enumerate([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]):
                    O_mat[i, base+k] = 0.5*(1.0 + np.dot(dvec, ud))

        # 1) Incremental KKT/IFT
        lam_inc = inc_opt.update(O_mat, g_const)
        AoI_inc.append(compute_aoi_objective(lam_inc, mu, O_mat, g_const))

        # 2) Proportional
        w        = O_mat.dot(g_const)
        lam_prop = (Lambda_max/num_cams if w.sum()<=EPS
                    else Lambda_max*(w/w.sum()))
        AoI_prop.append(compute_aoi_objective(lam_prop, mu, O_mat, g_const))

        # 3) Full PSO+SLSQP
        lam_full = hybrid_opt(compute_aoi_objective, mu,
                              O_mat, g_const, top_k=top_k_for_slsqp)
        AoI_full.append(compute_aoi_objective(lam_full, mu, O_mat, g_const))



        print(f"Frame {t}: "
              f"inc={AoI_inc[-1]:.4f}, "
              f"prop={AoI_prop[-1]:.4f}, "
              f"full={AoI_full[-1]:.4f}, "
              )

    # Plot results
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(frames, AoI_inc,  label='Incremental')
        plt.plot(frames, AoI_prop, label='Proportional')
        plt.plot(frames, AoI_full, label='PSO+SLSQP')

        plt.xlabel('Frame')
        plt.ylabel('AoI')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print('Skipping plot: matplotlib unavailable')
