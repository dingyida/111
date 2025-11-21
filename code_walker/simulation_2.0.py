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
data_folder              = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\7walker"
EPS                      = 1e-8

# Fixed λ (from your provided optimal)
lam_fixed = np.array([10.0, 1.0, 7.0 , 1.0,
                      6.0, 2.0, 3.0 , 1.0])

# ================= HELPERS =====================
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
    return pi0, pii0, pii1  # note: returns (pi0, pii_i0, pii_i1)

def theoretical_quantities(lam, mu, O_mat, g):
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    busy_frac = pii0 + pii1  # per-camera
    phi = mu * busy_frac + EPS  # effective completion rates
    vis_bin = (O_mat > 0).astype(float)
    P = vis_bin.T.dot(busy_frac)          # per-feature
    Phi = O_mat.T.dot(phi)                # per-feature
    L = lam.sum()
    DPj =  (vis_bin.T.dot(busy_frac) + EPS) / vis_bin.T.dot(phi)

    S = O_mat.T.dot(mu * busy_frac)  # denominator term: sum_i (pii0+pii1)*O_ij*mu_i
    AoIvec = g * (1.0 / (S + EPS) + 1.0 / L + DPj)
    return {
        "pi0": pi0,
        "pii0": pii0,
        "pii1": pii1,
        "busy_frac": busy_frac,
        "phi": phi,
        "vis_bin": vis_bin,
        "P": P,
        "Phi": Phi,
        "AoIvec": AoIvec,
        "L": L,
    }

def theoretical_quantities2(lam, mu, O_mat, g):
    """
    Compute theoretical quantities and AoI_j = g_j * (E[Y^2]_j * Phi_j + 1/L + DPj),
    with all uses of Lambda replaced by L = sum(lam).
    """
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
    q = numer_q / (L  )  # per-face

    # moments of L
    E_L = 1.0 / (q  )
    Var_L = 2*(1 - q) / (q*q)

    # B moments using L everywhere
    E_B = 1.0 / bar_mu + bar_mu / (L * (L + bar_mu) + EPS)
    E_B2 = 2.0 / (bar_mu ** 2 + EPS) + 2.0 / (L * (L + bar_mu) + EPS)

    # E[Y^2] per face
    EY2 = E_L * E_B2 + Var_L * (E_B ** 2)

    # custom AoI
    AoI_custom = g * (EY2 * Phi/2 + 1.0 / L + DPj)
    #AoI_custom = g * (EY2 / (2 * E_L * E_B) + 1.0 / L + DPj)
    return {
        "pi0": pi0,
        "pii0": pii0,
        "pii1": pii1,
        "busy_frac": busy_frac,
        "phi": phi,
        "vis_bin": vis_bin,
        "Phi": Phi,
        "DPj": DPj,
        "L": L,
        "bar_mu": bar_mu,
        "q": q,
        "E_L": E_L,
        "Var_L": Var_L,
        "E_B": E_B,
        "E_B2": E_B2,
        "EY2": EY2,
        "AoI_custom": AoI_custom,
    }


def simulate_and_empirical(lam, mu, O_mat, g, T=2000.0, seed=42):
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
    busy_empty_time = np.zeros(num_cams)   # serving, waiting slot empty
    busy_waiting_time = np.zeros(num_cams) # serving, waiting slot occupied

    # For Face 0 AoI trajectory (optional)
    face0_times = [0.0]
    face0_aoi = [0.0]

    while events:
        t, _, etype, info = heapq.heappop(events)
        if t > T:
            t = T
        dt = t - last_t
        # update empirical occupancy
        if processing is None:
            idle_time += dt
        else:
            cam_p = processing['cam']
            if waiting is None:
                busy_empty_time[cam_p] += dt
            else:
                busy_waiting_time[cam_p] += dt

        # AoI continuous evolution: integrate
        total_area += aoi * dt + 0.5 * g * (dt ** 2)
        aoi += g * dt
        last_t = t

        # track face 0
        face0_times.append(t)
        face0_aoi.append(aoi[0])

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
            # AoI reset per-feature with probability O_mat[cam,k]
            for k in range(num_feat):
                if random.random() <= O_mat[cam, k]:
                    aoi[k] = g[k] * delay

            # record after reset for face0
            face0_times.append(t)
            face0_aoi.append(aoi[0])

            if waiting is not None:
                nxt = waiting; waiting = None
                processing = nxt
                ptime = random.expovariate(mu[nxt['cam']])
                heapq.heappush(events, (t + ptime, evt_id, 'completion', processing.copy()))
                evt_id += 1
            else:
                processing = None

        if last_t >= T:
            break

    # Final stretch if last event < T (already accounted above by limiting t)
    emp_aoi = total_area / T
    emp_mean = np.mean(emp_aoi)

    # Empirical steady-state estimates
    pi0_emp = idle_time / T
    pii0_emp = busy_empty_time / T
    pii1_emp = busy_waiting_time / T
    busy_frac_emp = pii0_emp + pii1_emp
    phi_emp = mu * busy_frac_emp + EPS
    P_emp = (O_mat > 0).astype(float).T.dot(busy_frac_emp)
    Phi_emp = O_mat.T.dot(phi_emp)
    L = lam.sum()


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
        "events": events,
        "face0_times": face0_times,
        "face0_aoi": face0_aoi,
    }

# ================== MAIN ==================
if __name__ == "__main__":
    # load frame 0 and build O_mat
    files = sorted(glob.glob(os.path.join(data_folder, "frame_*.json")))
    if len(files) == 0:
        raise RuntimeError(f"No frame_*.json in {data_folder}")
    sample = json.load(open(files[0]))
    cams = sorted(c["name"] for c in sample["cameras"])
    num_cams = len(cams)

    # construct μ
    np.random.seed(93)
    mu = np.random.uniform(0.5, 2, size=num_cams)

    # build object list
    all_pids = sorted({
        pid
        for fp in files
        for c in json.load(open(fp))["cameras"]
        for pid in c.get("visible_ids", [])
    })
    num_objs = len(all_pids)

    # compute O_mat at frame 0 (faces)
    axes = {c["name"]: euler_to_vector(c["rotation"]["pitch"], c["rotation"]["yaw"]) for c in sample["cameras"]}
    vis = np.zeros((num_cams, num_objs), dtype=bool)
    for i, name in enumerate(cams):
        entry = next(c for c in sample["cameras"] if c["name"] == name)
        for pid in entry["visible_ids"]:
            j = all_pids.index(pid)
            vis[i, j] = True

    O_mat = np.zeros((num_cams, 4 * num_objs))
    directions = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    for i, name in enumerate(cams):
        dvec = axes[name]
        for j in range(num_objs):
            if not vis[i, j]:
                continue
            base = 4 * j
            for k, ud in enumerate(directions):
                O_mat[i, base + k] = 0.5 * (1.0 + np.dot(dvec, ud))

    # g per face
    np.random.seed(123)
    g = np.random.uniform(0.5, 2, size=4 * num_objs)

    # Theoretical quantities with fixed λ
    th = theoretical_quantities(lam_fixed, mu, O_mat, g)
    #th2 = theoretical_quantities2(lam_fixed, mu, O_mat, g)
    #print("Custom AoI mean:", np.mean(th2["AoI_custom"]))
    #Phi_vec = th2["Phi"]
    #EY2_vec = th2["EY2"]
    #q_vec = th2["q"]
    #E_L_vec = th2["E_L"]
    #Var_L_vec = th2["Var_L"]
    #E_B = th2["E_B"]  # scalar
    #E_B2 = th2["E_B2"]  # scalar
    #L = th2["L"]
    # Simulation empirical
    emp = simulate_and_empirical(lam_fixed, mu, O_mat, g, T=30000.0, seed=23)

    # ========== Print summary ==========
    #print("=== Per-face quantities (first 10 shown) ===")
    #print(f"Phi_j:              {Phi_vec[:10]}")
    #print(f"E[Y^2]_j:           {EY2_vec[:30]}")
    #print(f"q_j:                {q_vec[:30]}")
    #print(f"E[L]_j:             {E_L_vec[:10]}")
    #print(f"Var(L)_j:           {Var_L_vec[:10]}")
    #print()

    #print("=== Aggregated statistics ===")
    #print(f"Mean Phi_j:         {np.mean(Phi_vec):.6f}")
    #print(f"Mean E[Y^2]_j:      {np.mean(EY2_vec):.6f}")
    #print(f"Mean q_j:           {np.mean(q_vec):.6f}")
    #print(f"Mean E[L]_j:        {np.mean(E_L_vec):.6f}")
    #print(f"Mean Var(L)_j:      {np.mean(Var_L_vec):.6f}")
    #print(f"E[B]:              {E_B:.6f}")
    #print(f"E[B^2]:            {E_B2:.6f}")
    #print(f"L = sum(lambda):    {L:.6f}")
    print(f"\nEmpirical mean AoI (per-face average): {emp['emp_mean']:.6f}")
    mean_AoIvec_th = np.mean(th["AoIvec"])
    print("Mean AoIvec_j (theoretical):", mean_AoIvec_th)
    print("mu:", mu)
    print("=== Fixed λ_i ===")
    print("λ_i:", lam_fixed)
    print("\n=== Theoretical steady-state ===")
    print(f"π0 = {th['pi0']:.6f}")
    print(f"πi0 = {th['pii0']}")
    print(f"πi1 = {th['pii1']}")
    print(f"busy_frac (πi0+πi1) = {th['busy_frac']}")


    print("AoIvec_j (theoretical):", th["AoIvec"])








    # print("\n=== Empirical estimates from simulation ===")



    #print(f"π0_emp = {emp['pi0_emp']:.6f}")
    #print(f"πi0_emp = {emp['pii0_emp']}")
    #print(f"πi1_emp = {emp['pii1_emp']}")
    #print(f"busy_frac_emp = {emp['busy_frac_emp']}")


    #print("\nO_{i,k} matrix (face-level, camera i to face k):")
    #print(O_mat)

    # Optional: plot AoI evolution for face 0
    #plt.figure()
    #plt.plot(emp["face0_times"], emp["face0_aoi"], label="Face 0 AoI")
    #plt.xlabel("Time")
    #plt.ylabel("AoI (Face 0)")
    #plt.title("AoI Evolution for Face 0")
    #plt.grid(True)
    #plt.tight_layout()
    #plt.legend()
    #plt.show()
