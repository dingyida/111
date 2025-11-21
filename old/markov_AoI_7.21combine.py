import os
import glob
import json
import math
import numpy as np
from scipy.optimize import minimize

# ================= USER SETTINGS =================
data_folder   = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\output"
Lambda_max    = 10.0
EPS           = 1e-6
penalty_coef  = 1e3    # coefficient for linear penalty term in PSO
large_change_threshold = 0.15  # threshold for triggering full reoptimization
recompute_hessian_every = 20   # safety: force Hessian rebuild after this many increments

# ================== BASE HELPERS ==================

def walker_idx_to_pid(idx: int) -> int:
    return 24 + 2 * idx


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
            avg[pid] = 0.0; continue
        dist, dt = 0.0, 0
        for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
            dist += np.linalg.norm(p1 - p0)
            dt   += f1 - f0
        avg[pid] = dist / dt if dt > 0 else 0.0
    return avg


def euler_to_vector(pitch, yaw):
    p, y = math.radians(pitch), math.radians(yaw)
    return np.array([math.cos(p)*math.cos(y), math.cos(p)*math.sin(y), math.sin(p)])

# ================== AOI MODEL ==================

def compute_steady_state(lam, mu):
    L = lam.sum()
    A = np.sum(lam / (L + mu))
    B = np.sum(lam / (mu * (L + mu)))
    pi0  = (1 - A) / (1 + L * B)
    pii0 = pi0 * lam / ((L + mu) * (1 - A))
    pii1 = pii0 * (L / mu)
    return pi0, pii0, pii1


def compute_aoi_objective(lam, mu, A_mat, g):
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    phi = mu * (pii0 + pii1) + EPS        # avoid zero
    L   = lam.sum()
    pit = pii0 + pii1
    D   = np.zeros_like(mu)
    for i in range(len(mu)):
        t1 = pi0 * lam[i] / mu[i]
        t2 = sum(pit[k] * lam[i] * (mu[k] / (mu[k] + L)) *
                 (1/(mu[k] + L) + 1/mu[i]) for k in range(len(mu)))
        D[i] = (t1 + t2) / phi[i]
    mask   = A_mat.sum(0) > 0
    A_vis  = A_mat[:, mask]; g_vis = g[mask]
    Phi    = A_vis.T.dot(phi)
    Num    = A_vis.T.dot(phi * D)
    AoIvec = (Num / Phi) + 1.0 / Phi
    return float(g_vis.dot(AoIvec))


def gradient_lambda(lam, mu, A_mat, g, eps=1e-8):
    base = compute_aoi_objective(lam, mu, A_mat, g)
    grad = np.zeros_like(lam)
    for i in range(len(lam)):
        step = eps * max(1.0, abs(lam[i]))
        lam2  = lam.copy(); lam2[i] += step
        grad[i] = (compute_aoi_objective(lam2, mu, A_mat, g) - base) / step
    return grad

# ================== GLOBAL OPT (PSO+SLSQP) ==================

def pso_optimize(obj, mu, A_mat, g, swarm_size=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    N    = len(mu)
    X    = np.clip(np.random.rand(swarm_size, N), EPS, None)
    V    = np.zeros_like(X)
    pbest, pval = X.copy(), np.full(swarm_size, np.inf)
    gbest_val   = np.inf
    def penalized(lam):
        loss = obj(lam, mu, A_mat, g)
        cons = max(0.0, lam.sum() - Lambda_max)
        return loss + penalty_coef * cons
    for i in range(swarm_size):
        pval[i] = penalized(X[i])
        gbest_val = min(gbest_val, pval[i])
    for _ in range(max_iter):
        gbest_i = np.argmin(pval)
        for i in range(swarm_size):
            r1, r2 = np.random.rand(N), np.random.rand(N)
            V[i]   = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(X[gbest_i]-X[i])
            X[i]  += V[i]; X[i] = np.clip(X[i], EPS, None)
            v      = penalized(X[i])
            if v < pval[i]: pval[i], pbest[i] = v, X[i].copy();
            gbest_val = min(gbest_val, pval[i])
    return pbest, pval


def hybrid_opt(obj, mu, A_mat, g, top_k=10):
    pbest, pval = pso_optimize(obj, mu, A_mat, g)
    idxs        = np.argsort(pval)[:top_k]
    bounds      = [(EPS, Lambda_max)]*len(mu)
    cons        = {'type':'eq', 'fun': lambda lam: lam.sum()-Lambda_max}
    best, val   = None, np.inf
    for i in idxs:
        res = minimize(obj, pbest[i], args=(mu, A_mat, g), method='SLSQP',
                       bounds=bounds, constraints=[cons], options={'ftol':1e-6,'maxiter':50})
        if res.fun < val: best, val = res.x.copy(), res.fun
    return best

# ================== INCREMENTAL UPDATE (KKT / IFT) ==================

class IncrementalOptimizer:
    def __init__(self, mu, Lambda_max, use_fd=True):
        self.mu     = mu; self.Lambda_max = Lambda_max
        self.use_fd = use_fd
        self.lam = None; self.nu = 0.0; self.H = None
        self.count = 0
        self.prev_A = None; self.prev_g = None

    def _project(self, lam):
        lam = np.maximum(lam, EPS)
        lam *= self.Lambda_max / max(lam.sum(), EPS)
        return lam

    def build_hessian(self, lam, A_mat, g):
        N     = len(lam)
        H     = np.zeros((N,N))
        grad0 = gradient_lambda(lam, self.mu, A_mat, g)
        for i in range(N):
            lam2 = lam.copy(); lam2[i] += EPS
            grad2 = gradient_lambda(lam2, self.mu, A_mat, g)
            H[:,i] = (grad2 - grad0)/EPS
        return 0.5*(H+H.T)

    def initialize(self, A_mat, g):
        self.lam = hybrid_opt(compute_aoi_objective, self.mu, A_mat, g)
        self.lam = self._project(self.lam)
        grad     = gradient_lambda(self.lam, self.mu, A_mat, g)
        self.nu  = grad.mean()
        self.H   = self.build_hessian(self.lam, A_mat, g)
        self.count=0; self.prev_A, self.prev_g = A_mat.copy(), g.copy()

    def update(self, A_mat, g):
        if self.lam is None:
            self.initialize(A_mat, g); return self.lam
        # small-change test
        relA = np.linalg.norm(A_mat-self.prev_A)/(np.linalg.norm(self.prev_A)+EPS)
        relg = np.linalg.norm(g-self.prev_g)/(np.linalg.norm(self.prev_g)+EPS)
        if relA>large_change_threshold or relg>large_change_threshold or self.count>=recompute_hessian_every:
            self.initialize(A_mat, g); return self.lam
        # solve KKT sensitivity
        grad_old = gradient_lambda(self.lam, self.mu, self.prev_A, self.prev_g)
        grad_new = gradient_lambda(self.lam, self.mu, A_mat, g)
        g_lO     = grad_new - grad_old
        N        = len(self.lam); o = np.ones(N)
        K        = np.block([[self.H, -o[:,None]], [o[None,:], np.zeros((1,1))]])
        rhs      = np.concatenate([-g_lO, [0.0]])
        try: d = np.linalg.solve(K, rhs)
        except: self.initialize(A_mat, g); return self.lam
        self.lam = self._project(self.lam + d[:N]); self.nu += d[N]
        self.count+=1; self.prev_A, self.prev_g = A_mat.copy(), g.copy()
        return self.lam

# ================== MAIN LOOP ==================

if __name__ == "__main__":
    avg_speeds    = compute_average_speeds(data_folder)
    avg_mean      = np.mean(list(avg_speeds.values()))
    sample        = json.load(open(os.path.join(data_folder,'frame_00000.json')))
    cams          = sorted(c['name'] for c in sample['cameras'])
    num_cams      = len(cams)
    obs_dirs      = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]])
    files         = sorted(glob.glob(os.path.join(data_folder,'frame_*.json')))
    frames        = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
    mu            = np.random.uniform(0.5,2.0,size=num_cams)

    inc_opt       = IncrementalOptimizer(mu, Lambda_max)
    lam_uniform   = np.ones(num_cams)*(Lambda_max/num_cams)

    AoI_inc, AoI_u, AoI_full = [], [], []

    for fp, t in zip(files, frames):
        data   = json.load(open(fp))
        axes   = {c['name']: euler_to_vector(c['rotation']['pitch'], c['rotation']['yaw']) for c in data['cameras']}
        pids   = sorted({pid for c in data['cameras'] for pid in c['visible_ids']})
        speeds = np.array([avg_speeds.get(pid, avg_mean) for pid in pids])
        g_t    = np.repeat(speeds, 4)
        # build visibility & A_mat
        vis    = np.zeros((num_cams,len(pids)),bool)
        for i,name in enumerate(cams):
            entry = next(c for c in data['cameras'] if c['name']==name)
            for pid in entry['visible_ids']:
                vis[i,pids.index(pid)] = True
        A_mat  = np.zeros((num_cams,4*len(pids)))
        for i,name in enumerate(cams):
            d = axes[name]
            for j in range(len(pids)):
                if not vis[i,j]: continue
                for k,ud in enumerate(obs_dirs):
                    A_mat[i,4*j+k] = 0.5*(1+d.dot(ud))
        # Strategy 1: Incremental
        lam_inc    = inc_opt.update(A_mat, g_t)
        val_inc    = compute_aoi_objective(lam_inc, mu, A_mat, g_t)
        AoI_inc.append(val_inc)
        # Strategy 2: Uniform
        val_unif   = compute_aoi_objective(lam_uniform, mu, A_mat, g_t)
        AoI_u.append(val_unif)
        # Strategy 3: PSO+SLSQP full
        lam_full   = hybrid_opt(compute_aoi_objective, mu, A_mat, g_t)
        val_full   = compute_aoi_objective(lam_full, mu, A_mat, g_t)
        AoI_full.append(val_full)
        print(f"Frame {t}: inc={val_inc:.4f}, unif={val_unif:.4f}, full={val_full:.4f}")

    # Plot comparison
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(frames, AoI_inc,  label='Incremental')
        plt.plot(frames, AoI_u,    label='Uniform')
        plt.plot(frames, AoI_full, label='PSO+SLSQP')
        plt.xlabel('Frame'); plt.ylabel('AoI'); plt.legend(); plt.show()
    except ImportError:
        print('Skipping plot: matplotlib unavailable')
