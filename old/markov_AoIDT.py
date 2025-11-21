import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from typing import Tuple

# ---------------------
# 1. Original CTMC Methods
# ---------------------
def compute_steady_state_expanded(lambda_arrivals: np.ndarray, mu: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    a = lambda_arrivals.size
    Λ = np.sum(lambda_arrivals)
    A_val = np.sum(lambda_arrivals / (Λ + mu))
    B_val = np.sum(lambda_arrivals / (mu * (Λ + mu)))
    pi_00 = (1.0 - A_val) / (1.0 + Λ * B_val)
    pi_i0 = pi_00 * lambda_arrivals / ((Λ + mu) * (1.0 - A_val))
    pi_i1_agg = pi_00 * (lambda_arrivals * Λ) / (mu * (Λ + mu) * (1.0 - A_val))
    pi_ij = np.outer(pi_i1_agg, lambda_arrivals / Λ)
    return pi_00, pi_i0, pi_ij

def compute_aoi_original(A_mat: np.ndarray, lambda_arrivals: np.ndarray, mu: np.ndarray,
                         pi_00: float, pi_i0: np.ndarray, pi_ij: np.ndarray,
                         g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b = A_mat.shape
    Λ = np.sum(lambda_arrivals)
    # Throughput per camera
    phi = mu * (pi_i0 + pi_ij.sum(axis=1))
    # Compute conditional delays
    served_probs = pi_i0 + pi_ij.sum(axis=1)
    denom = pi_00 + np.sum(served_probs * (mu / (mu + Λ)))
    D = np.zeros(a)
    for i in range(a):
        term0 = pi_00 / mu[i]
        termk = np.sum(served_probs * (mu / (mu + Λ)) * (1/(mu + Λ) + 1/mu[i]))
        D[i] = (term0 + termk) / denom
    # Update rates and AoI per object
    Phi = (phi[:, None] * A_mat).sum(axis=0)
    AoI = np.zeros(b)
    for j in range(b):
        if Phi[j] > 0:
            weights = (phi * A_mat[:, j]) / Phi[j]
            AoI[j] = g[j] * (1/Phi[j] + np.dot(weights, D))
        else:
            AoI[j] = np.nan
    return phi, D, Phi, AoI

# Objective wrapper for original method (sum lambda = 1)
def total_aoi_original(lambdas: np.ndarray) -> float:
    pi_00, pi_i0, pi_ij = compute_steady_state_expanded(lambdas, mu)
    _, _, _, AoI = compute_aoi_original(A, lambdas, mu, pi_00, pi_i0, pi_ij, g)
    return np.nansum(AoI)


# ---------------------
# 2. Simplified (Λ = 1) Methods
# ---------------------
def compute_simple_metrics(P: np.ndarray, mu: np.ndarray,
                           A_mat: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # A and B under Lambda=1
    A_val = np.sum(P / (1 + mu))
    B_val = np.sum(P / (mu * (1 + mu)))
    # Vectorized conditional delay D
    E = (1 - A_val) + np.dot(P, mu/(mu + 1))
    C = np.dot(P, mu/(mu + 1)**2)
    D = C/E + 1.0/mu
    # Throughput per camera
    phi = P / (1 + B_val)
    # Update rates and AoI per object
    Phi = (phi[:, None] * A_mat).sum(axis=0)
    AoI = np.zeros_like(g)
    for j in range(g.size):
        if Phi[j] > 0:
            weights = phi * A_mat[:, j] / Phi[j]
            AoI[j] = g[j] * (1.0/Phi[j] + np.dot(weights, D))
        else:
            AoI[j] = np.nan
    return phi, D, Phi, AoI

# Objective wrapper for simplified method (sum P = 1)
def total_aoi_simple(P: np.ndarray) -> float:
    _, _, _, AoI = compute_simple_metrics(P, mu, A, g)
    return np.nansum(AoI)


# ---------------------
# 3. Generate Example Data
# ---------------------
np.random.seed(42)
a = b = 100
mask = np.random.rand(a, b) < 0.5
values = np.random.rand(a, b)
A = np.where(mask, 0, values)  # Random incidence weights
mu = np.random.uniform(1.0, 2.0, size=a)
g = np.random.uniform(0.8, 1.2, size=b)

# ---------------------
# 4. Optimization Setup
# ---------------------
cons_eq = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1)] * a
x0 = np.full(a, 1.0/a)

# Optimize original CTMC objective
start_orig = time.perf_counter()
res_orig = minimize(total_aoi_original, x0, method='SLSQP',
                    bounds=bounds, constraints=cons_eq)
time_orig = time.perf_counter() - start_orig
P_orig = res_orig.x
phi_orig, D_orig, Phi_orig, AoI_orig = compute_aoi_original(A, P_orig, mu,
                                                           *compute_steady_state_expanded(P_orig, mu), g)
total_AoI_orig = np.nansum(AoI_orig)

# Optimize simplified objective
start_simp = time.perf_counter()
res_simp = minimize(total_aoi_simple, x0, method='SLSQP',
                    bounds=bounds, constraints=cons_eq)
time_simp = time.perf_counter() - start_simp
P_simp = res_simp.x
phi_simp, D_simp, Phi_simp, AoI_simp = compute_simple_metrics(P_simp, mu, A, g)
total_AoI_simp = np.nansum(AoI_simp)

# ---------------------
# 5. Display Results
# ---------------------
# Camera-level comparison
df_cam = pd.DataFrame({
    "P_orig": P_orig,
    "P_simp": P_simp,
    "D_orig": D_orig,
    "D_simp": D_simp
}, index=[f"Cam {i+1}" for i in range(a)])

# Object-level comparison
df_obj = pd.DataFrame({
    "g_j": g,
    "Phi_orig": Phi_orig,
    "Phi_simp": Phi_simp,
    "AoI_orig": AoI_orig,
    "AoI_simp": AoI_simp
}, index=[f"Obj {j+1}" for j in range(b)])

# Summary: total AoI and computation time
df_summary = pd.DataFrame({
    "Method": ["Original CTMC", "Simplified"],
    "Total_AoI": [total_AoI_orig, total_AoI_simp],
    "Comp_Time_s": [time_orig, time_simp]
})

print("\nCamera-Level Comparison:")
print(df_cam.to_string(float_format="{:.6f}".format))

print("\nObject-Level Comparison:")
print(df_obj.to_string(float_format="{:.6f}".format))

print("\nSummary:")
print(df_summary.to_string(index=False, float_format="{:.6f}".format))

