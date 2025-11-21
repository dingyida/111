import os
import glob
import json
import math
import heapq
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ================= USER SETTINGS =================
data_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\3walker"
Lambda_max = 40
EPS = 1e-8
penalty_coef = 1e3
swarm_size = 30
pso_max_iter = 100
slsqp_max_iter = 100
top_k_for_slsqp = 10

# ================== HELPERS =====================
def euler_to_vector(pitch_deg, yaw_deg):
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    return np.array([math.cos(p)*math.cos(y),
                     math.cos(p)*math.sin(y),
                     math.sin(p)])


# ================== AOI THEORY / SIM ==================
def compute_steady_state(lam, mu):
    L = lam.sum()
    A = np.sum(lam / (L + mu))
    B = np.sum(lam / (mu * (L + mu)))
    pi0 = (1 - A) / (1 + L * B)
    pii0 = pi0 * lam / ((L + mu) * (1 - A))
    pii1 = pii0 * (L / mu)
    return pi0, pii0, pii1

def theoretical_quantities(lam, mu, O_mat, g):
    L = lam.sum()

    # steady-state occupancy
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    busy_frac = pii0 + pii1  # per-camera busy probabilities
    bar_mu = np.sum(busy_frac * mu)  # weighted service rate

    # effective completion per camera
    phi = mu * busy_frac + EPS

    # visibility binary and aggregates
    vis_bin = (O_mat > 0).astype(float)
    Phi = O_mat.T.dot(phi)  # per-face weighted refresh rate
    DPj = vis_bin.T.dot(phi) / (vis_bin.T.dot(busy_frac) + EPS)  # per-face average phi

    # compute q_j per face
    numer_q = (lam[:, None] * O_mat).sum(axis=0)  # sum_i lambda_i * O_ij
    q = numer_q / (L)  # per-face

    # moments of L
    E_L = 1.0 / (q)
    E_LL = 2 * (1 - q) / (q * q)

    # B moments using L everywhere
    E_B = 1.0 / bar_mu + bar_mu / (L * (L + bar_mu) + EPS)
    E_B2 = 2.0 / (bar_mu**2 + EPS) + (2.0 * bar_mu * (2.0 * L + bar_mu)) / ((L**2) * (L + bar_mu)**2 + EPS)

    # E[Y^2] per face
    EY2 = E_L * E_B2 + E_LL * (E_B ** 2)

    # custom AoI
    AoIvec = g * (EY2 * Phi / 2 + 1.0 / L + DPj)


    return {
        "AoIvec": AoIvec,
        "busy_frac": busy_frac,
        "phi": phi,

        "Phi": Phi,
        "L": L,
        "pi0": pi0,
        "pii0": pii0,
        "pii1": pii1,
    }

def simulate_and_empirical(lam, mu, O_mat, g, T=4800.0, seed=42):
    random.seed(seed); np.random.seed(seed)
    num_cams = len(lam); num_feat = len(g)
    events = []
    evt_id = 0

    # Pre-generate Poisson arrivals per camera
    for i in range(num_cams):
        t = 0.0
        while True:
            dt = random.expovariate(lam[i])
            t += dt
            if t > T: break
            heapq.heappush(events, (t, evt_id, 'arrival', {'cam': i, 'arrival_time': t}))
            evt_id += 1

    processing = None
    waiting = None
    aoi = np.zeros(num_feat)
    total_area = np.zeros(num_feat)
    last_t = 0.0

    # Empirical occupancy time counters
    idle_time = 0.0
    busy_empty_time = np.zeros(num_cams)
    busy_waiting_time = np.zeros(num_cams)

    while events:
        t, _, etype, info = heapq.heappop(events)
        if t > T:
            break
        dt = t - last_t
        if processing is None:
            idle_time += dt
        else:
            cam_p = processing['cam']
            if waiting is None:
                busy_empty_time[cam_p] += dt
            else:
                busy_waiting_time[cam_p] += dt

        # AoI evolution
        total_area += aoi * dt + 0.5 * g * (dt ** 2)
        aoi += g * dt
        last_t = t

        if etype == 'arrival':
            cam = info['cam']; atime = info['arrival_time']
            if processing is None:
                processing = {'cam': cam, 'arrival_time': atime}
                ptime = random.expovariate(mu[cam])
                heapq.heappush(events, (t + ptime, evt_id, 'completion', processing.copy()))
                evt_id += 1
            else:
                waiting = {'cam': cam, 'arrival_time': atime}
        else:  # completion
            cam = info['cam']; atime = info['arrival_time']
            delay = t - atime
            for k in range(num_feat):
                if random.random() <= O_mat[cam, k]:
                    aoi[k] = g[k] * delay
            if waiting is not None:
                nxt = waiting; waiting = None
                processing = nxt
                ptime = random.expovariate(mu[nxt['cam']])
                heapq.heappush(events, (t + ptime, evt_id, 'completion', processing.copy()))
                evt_id += 1
            else:
                processing = None

    emp_aoi = total_area / T
    emp_mean = np.mean(emp_aoi)
    pi0_emp = idle_time / T
    pii0_emp = busy_empty_time / T
    pii1_emp = busy_waiting_time / T
    busy_frac_emp = pii0_emp + pii1_emp
    phi_emp = mu * busy_frac_emp + EPS
    P_emp = (O_mat > 0).astype(float).T.dot(busy_frac_emp)
    Phi_emp = O_mat.T.dot(phi_emp)
    return {
        "emp_mean": emp_mean,
        "emp_aoi_vec": emp_aoi,
        "pi0_emp": pi0_emp,
        "pii0_emp": pii0_emp,
        "pii1_emp": pii1_emp,
        "busy_frac_emp": busy_frac_emp,
        "phi_emp": phi_emp,
        "P_emp": P_emp,
        "Phi_emp": Phi_emp,
    }

# ================== OPTIMIZATION ==================
def pso_optimize(obj, mu, O_mat, g,
                 swarm_size=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    N = len(mu)
    X = np.clip(np.random.rand(swarm_size, N), EPS, None)
    V = np.zeros_like(X)
    pbest = X.copy()
    pval  = np.full(swarm_size, np.inf)

    def penalized(lam_local):
        loss = obj(lam_local, mu, O_mat, g)
        cons = max(0.0, lam_local.sum() - Lambda_max)
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
    idxs = np.argsort(pval)[:top_k]
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons = {'type': 'eq', 'fun': lambda lam: lam.sum() - Lambda_max}
    best, val = None, np.inf
    for i in idxs:
        res = minimize(obj, pbest[i], args=(mu, O_mat, g),
                       method='SLSQP', bounds=bounds,
                       constraints=[cons],
                       options={'ftol':1e-6, 'maxiter':slsqp_max_iter})
        if res.success and res.fun < val:
            best, val = res.x.copy(), res.fun
    if best is None:
        return np.ones_like(mu) * (Lambda_max / len(mu))
    return best

class IncrementalOptimizer:
    def __init__(self, mu, Lambda_max):
        self.mu = mu
        self.Lambda_max = Lambda_max
        self.lam = None

    def update(self, O_mat, g):
        if self.lam is None:
            self.lam = np.ones_like(self.mu) * (self.Lambda_max / len(self.mu))
        self.lam = hybrid_opt(lambda lam_local, mu_local, O_local, g_local: float(np.mean(theoretical_quantities(lam_local, self.mu, O_local, g_local)["AoIvec"])),
                              self.mu, O_mat, g, top_k=top_k_for_slsqp)
        self.lam = np.maximum(self.lam, EPS)
        self.lam *= self.Lambda_max / (self.lam.sum() + EPS)
        return self.lam

# ================== MAIN DRIVER ==================
def main():
    files = sorted(glob.glob(os.path.join(data_folder, "frame_*.json")))
    if not files:
        raise RuntimeError(f"No frame_*.json in {data_folder}")
    frames = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
    sample = json.load(open(files[0]))
    cams = sorted(c["name"] for c in sample["cameras"])
    num_cams = len(cams)

    np.random.seed(42)
    mu = np.random.uniform(0.5, 2.0, size=num_cams)
    np.random.seed(123)
    all_pids = sorted({
        pid
        for fp in files
        for c in json.load(open(fp))["cameras"]
        for pid in c.get("visible_ids", [])
    })
    g_const = np.random.uniform(0.5, 2.0, size=4 * len(all_pids))

    lambda_schemes = ["incremental", "proportional", "full"]
    n_frames = len(frames)
    lam_store = {scheme: np.zeros((n_frames, num_cams)) for scheme in lambda_schemes}
    emp_aoi = {scheme: np.zeros(n_frames) for scheme in lambda_schemes}
    theo_inc_aoi = np.zeros(n_frames)

    inc_opt = IncrementalOptimizer(mu, Lambda_max)

    for idx, (fp, t) in enumerate(zip(files, frames)):
        data = json.load(open(fp))
        axes = {c["name"]: euler_to_vector(c["rotation"]["pitch"], c["rotation"]["yaw"])
                for c in data["cameras"]}
        vis = np.zeros((num_cams, len(all_pids)), dtype=bool)
        for i, name in enumerate(cams):
            entry = next(c for c in data["cameras"] if c["name"] == name)
            for pid in entry["visible_ids"]:
                j = all_pids.index(pid)
                vis[i, j] = True

        O_mat = np.zeros((num_cams, 4 * len(all_pids)))
        directions = [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]
        for i, name in enumerate(cams):
            dvec = axes[name]
            for j in range(len(all_pids)):
                if not vis[i,j]:
                    continue
                base = 4 * j
                for k, ud in enumerate(directions):
                    O_mat[i, base + k] = 0.5 * (1.0 + np.dot(dvec, ud))

        # λ schemes
        lam_inc = inc_opt.update(O_mat, g_const)
        lam_store["incremental"][idx] = lam_inc

        w_prop = O_mat.dot(g_const)
        if w_prop.sum() <= EPS:
            lam_prop = np.ones_like(mu) * (Lambda_max / len(mu))
        else:
            lam_prop = Lambda_max * (w_prop / w_prop.sum())
        lam_store["proportional"][idx] = lam_prop

        def obj_full(lam_local, mu_local, O_local, g_local):
            th = theoretical_quantities(lam_local, mu_local, O_local, g_local)
            return float(np.mean(th["AoIvec"]))
        lam_full = hybrid_opt(obj_full, mu, O_mat, g_const, top_k=top_k_for_slsqp)
        lam_store["full"][idx] = lam_full

        # empirical AoIs with T=4800 and different seeds
        emp_inc = simulate_and_empirical(lam_inc, mu, O_mat, g_const, T=4800.0, seed=100 + idx)
        emp_prop = simulate_and_empirical(lam_prop, mu, O_mat, g_const, T=4800.0, seed=1000 + idx)
        emp_full = simulate_and_empirical(lam_full, mu, O_mat, g_const, T=4800.0, seed=5000 + idx)

        emp_aoi["incremental"][idx] = emp_inc["emp_mean"]
        emp_aoi["proportional"][idx] = emp_prop["emp_mean"]
        emp_aoi["full"][idx] = emp_full["emp_mean"]

        th_inc = theoretical_quantities(lam_inc, mu, O_mat, g_const)
        theo_inc_aoi[idx] = np.mean(th_inc["AoIvec"])

        print(f"[Frame {t}] Empirical AoI: inc={emp_aoi['incremental'][idx]:.4f}, prop={emp_aoi['proportional'][idx]:.4f}, full={emp_aoi['full'][idx]:.4f}; Theo-inc={theo_inc_aoi[idx]:.4f}")

    # Averages across frames
    avg_emp_inc = np.mean(emp_aoi["incremental"])
    avg_emp_prop = np.mean(emp_aoi["proportional"])
    avg_emp_full = np.mean(emp_aoi["full"])
    avg_theo_inc = np.mean(theo_inc_aoi)

    print("\n=== Average over all frames ===")
    print(f"Empirical Incremental AoI: {avg_emp_inc:.6f}")
    print(f"Empirical Proportional AoI: {avg_emp_prop:.6f}")
    print(f"Empirical Full AoI: {avg_emp_full:.6f}")
    print(f"Theoretical Incremental AoI: {avg_theo_inc:.6f}")

    # Save results
    out_path = os.path.join(os.path.dirname(data_folder), "lambda_aoi_reduced.npz")
    np.savez_compressed(out_path,
                        frames=np.array(frames),
                        lam_store=lam_store,
                        emp_aoi=emp_aoi,
                        theo_inc_aoi=theo_inc_aoi,
                        lambda_schemes=lambda_schemes)
    print("Saved reduced results to", out_path)
    return out_path

if __name__ == "__main__":
    result_file = main()
    print("Result file:", result_file)
