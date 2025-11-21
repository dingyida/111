import os
import glob
import json
import math
import numpy as np
from scipy.optimize import minimize
import heapq

# ================= USER SETTINGS =================
data_folder = r"C:\Users\dyd\PycharmProjects\PythonProject\AOI_old\7walker"
SIM_TIME_MAX = 200.0
Lambda_max = 20
EPS = 1e-9

# Optimization settings
swarm_size = 30
pso_max_iter = 100
slsqp_max_iter = 100
top_k_for_slsqp = 10
penalty_coef = 1e3


# ================== HELPER AND AOI MODEL (from previous code) ==================
def euler_to_vector(pitch_deg, yaw_deg):
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    return np.array([math.cos(p) * math.cos(y),
                     math.cos(p) * math.sin(y),
                     math.sin(p)])


def compute_steady_state(lam, mu):
    L = lam.sum()
    if L < EPS: L = EPS
    A = np.sum(lam / (L + mu))
    B = np.sum(lam / (mu * (L + mu)))
    pi0 = (1 - A) / (1 + L * B) if (1 + L * B) > EPS else 0.0
    pii0 = pi0 * lam / ((L + mu) * (1 - A)) if (1 - A) > EPS else np.zeros_like(lam)
    pii1 = pii0 * (L / mu)
    return pi0, pii0, pii1


def compute_aoi_objective(lam, mu, O_mat, g):
    pi0, pii0, pii1 = compute_steady_state(lam, mu)
    L = lam.sum()
    if L < EPS: return 1e9
    phi = mu * (pii0 + pii1)
    Phi = O_mat.T.dot(phi)
    pit = pii0 + pii1
    visibility_mask = O_mat > EPS
    P = visibility_mask.T.dot(pit)
    Phi = np.maximum(Phi, EPS)
    AoIvec = (1.0 / L) + (1.0 + 2.0 * P) / Phi
    obj_mask = O_mat.sum(0) > 0
    return float(g[obj_mask].dot(AoIvec[obj_mask]))


def pso_optimize(obj, mu, O_mat, g, swarm_size, max_iter, w=0.5, c1=1.5, c2=1.5):
    N = len(mu)
    X = np.clip(np.random.rand(swarm_size, N) * Lambda_max, EPS, Lambda_max)
    V = np.zeros_like(X)
    pbest, pval = X.copy(), np.full(swarm_size, np.inf)

    def penalized(lam):
        loss = obj(lam, mu, O_mat, g)
        cons = max(0.0, lam.sum() - Lambda_max)
        return loss + penalty_coef * cons

    for i in range(swarm_size):
        pval[i] = penalized(X[i])
    gbest_i = np.argmin(pval)

    for _ in range(max_iter):
        for i in range(swarm_size):
            r1, r2 = np.random.rand(N), np.random.rand(N)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (X[gbest_i] - X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], EPS, Lambda_max)
            v = penalized(X[i])
            if v < pval[i]:
                pval[i], pbest[i] = v, X[i].copy()
                if v < pval[gbest_i]:
                    gbest_i = i
    return pbest, pval


def hybrid_opt(obj, mu, O_mat, g, top_k):
    pbest, pval = pso_optimize(obj, mu, O_mat, g, swarm_size, pso_max_iter)
    idxs = np.argsort(pval)[:top_k]
    bounds = [(EPS, Lambda_max)] * len(mu)
    cons = {'type': 'eq', 'fun': lambda lam: lam.sum() - Lambda_max}
    best_res = None

    for i in idxs:
        res = minimize(obj, pbest[i], args=(mu, O_mat, g), method='SLSQP', bounds=bounds, constraints=[cons],
                       options={'ftol': 1e-7, 'maxiter': slsqp_max_iter})
        if res.success and (best_res is None or res.fun < best_res.fun):
            best_res = res

    return best_res.x if best_res is not None else pbest[np.argmin(pval)]


# ================== MAIN SIMULATION ==================
if __name__ == "__main__":
    np.random.seed(43)  # for reproducibility

    # 1. SETUP: Load data for frame 0 to get the fixed system parameters
    print("--- 1. System Setup ---")
    files = sorted(glob.glob(os.path.join(data_folder, 'frame_*.json')))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_folder}")

    frame0_file = files[0]
    data = json.load(open(frame0_file))

    cams = sorted([c['name'] for c in data['cameras'] if 'walker' in c['name']])
    num_cams = len(cams)

    # Use a fixed number of objects as specified
    num_objects = 7

    # Create the fixed Observation Matrix O_mat based on frame 0
    # We will generate a random one for this example since visibility logic can be complex
    print(f"Assuming {num_objects} objects and generating a random visibility matrix O_mat.")
    O_mat = np.random.rand(num_cams, num_objects)
    O_mat[O_mat < 0.7] = 0  # Make it sparse

    # Generate random mu and g vectors
    mu = np.random.uniform(0.5, 2.0, size=num_cams)
    g = np.random.uniform(0.5, 2.0, size=num_objects)
    aoi_rate = g

    print(f"Setup complete. num_cams={num_cams}, num_objects={num_objects}")

    # 2. ANALYTICAL: Run optimization to get optimal lambda and analytical AoI
    print("\n--- 2. Calculating Analytical AoI ---")
    lam_opt = hybrid_opt(compute_aoi_objective, mu, O_mat, g, top_k_for_slsqp)
    analytical_aoi = compute_aoi_objective(lam_opt, mu, O_mat, g)
    print(f"Optimal Lambda (λ): {np.round(lam_opt, 2)}")
    print(f"Analytical Mean AoI from formula: {analytical_aoi:.4f}")

    # 3. EVENT GENERATION: Pre-compute all frame arrivals
    print("\n--- 3. Pre-generating Frame Arrivals ---")
    arrivals = []
    for i in range(num_cams):
        if lam_opt[i] > EPS:
            t = 0
            while t < SIM_TIME_MAX:
                t += np.random.exponential(1.0 / lam_opt[i])
                if t < SIM_TIME_MAX:
                    arrivals.append({'arrival_time': t, 'cam_idx': i})

    arrivals.sort(key=lambda x: x['arrival_time'])
    print(f"Generated {len(arrivals)} total arrival events.")

    # 4. SIMULATION: Run the discrete-event simulation
    print("\n--- 4. Running Discrete-Event Simulation ---")
    current_time = 0.0
    aoi = np.zeros(num_objects)
    aoi_integral = np.zeros(num_objects)

    processing_slot = None
    waiting_slot = None

    event_queue = []  # A priority queue: (time, event_type, data)

    # Schedule all arrivals
    for frame in arrivals:
        heapq.heappush(event_queue, (frame['arrival_time'], 'arrival', frame))

    while event_queue:
        time, event_type, data = heapq.heappop(event_queue)

        if time > SIM_TIME_MAX:
            break

        # Update AoI linearly up to the event time
        delta_t = time - current_time
        aoi_integral += aoi * delta_t + 0.5 * aoi_rate * delta_t ** 2
        aoi += aoi_rate * delta_t
        current_time = time

        if event_type == 'arrival':
            if processing_slot is None:
                # Move directly to processing
                processing_slot = data
                proc_time = np.random.exponential(1.0 / mu[data['cam_idx']])
                finish_time = current_time + proc_time
                heapq.heappush(event_queue, (finish_time, 'departure', processing_slot))
            else:
                # Replace frame in waiting slot
                waiting_slot = data

        elif event_type == 'departure':
            finished_frame = data
            cam_idx = finished_frame['cam_idx']

            # Probabilistically reset AoI for each object
            for j in range(num_objects):
                if np.random.rand() < O_mat[cam_idx, j]:
                    delay = current_time - finished_frame['arrival_time']
                    aoi[j] = g[j] * delay

            # Move frame from waiting to processing
            if waiting_slot is not None:
                processing_slot = waiting_slot
                waiting_slot = None
                proc_time = np.random.exponential(1.0 / mu[processing_slot['cam_idx']])
                finish_time = current_time + proc_time
                heapq.heappush(event_queue, (finish_time, 'departure', processing_slot))
            else:
                processing_slot = None

    # Final AoI update to the end of the simulation
    if current_time < SIM_TIME_MAX:
        delta_t = SIM_TIME_MAX - current_time
        aoi_integral += aoi * delta_t + 0.5 * aoi_rate * delta_t ** 2

    # 5. RESULTS: Calculate and compare AoIs
    print("\n--- 5. Simulation Results ---")
    empirical_aoi_per_object = aoi_integral / SIM_TIME_MAX
    empirical_mean_aoi = np.mean(empirical_aoi_per_object)

    print(f"Empirical Time-Averaged AoI from simulation: {empirical_aoi_per_object}")
    print(f"Empirical Time-Averaged AoI from simulation: {empirical_mean_aoi}")