import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Original functions
def compute_steady_state_expanded(lambda_arrivals: np.ndarray, mu: np.ndarray):
    a = lambda_arrivals.size
    Λ = np.sum(lambda_arrivals)
    A_val = np.sum(lambda_arrivals / (Λ + mu))
    B_val = np.sum(lambda_arrivals / (mu * (Λ + mu)))
    pi_00 = (1.0 - A_val) / (1.0 + Λ * B_val)
    pi_i0 = pi_00 * lambda_arrivals / ((Λ + mu) * (1.0 - A_val))
    pi_i1_agg = pi_00 * (lambda_arrivals * Λ) / (mu * (Λ + mu) * (1.0 - A_val))
    pi_ij = np.outer(pi_i1_agg, lambda_arrivals / Λ)
    return pi_00, pi_i0, pi_ij

def compute_aoi(A_mat, lambda_arrivals, mu, pi_00, pi_i0, pi_ij, g):
    a, b = A_mat.shape
    Λ = np.sum(lambda_arrivals)
    phi = mu * (pi_i0 + pi_ij.sum(axis=1))
    served_probs = pi_i0 + pi_ij.sum(axis=1)
    denom = pi_00 + np.sum(served_probs * (mu / (mu + Λ)))
    D = np.zeros(a)
    for i in range(a):
        term0 = pi_00 / mu[i]
        termk = np.sum(served_probs * (mu / (mu + Λ)) * (1/(mu + Λ) + 1/mu[i]))
        D[i] = (term0 + termk) / denom
    Phi = (phi[:, None] * A_mat).sum(axis=0)
    AoI = np.zeros(b)
    for j in range(b):
        if Phi[j] > 0:
            weights = (phi * A_mat[:, j]) / Phi[j]
            AoI[j] = g[j] * (1/Phi[j] + np.dot(weights, D))
        else:
            AoI[j] = np.nan
    return phi, D, Phi, AoI

# Simplified functions (Λ = 1)
def compute_simple_A_B(P, mu):
    A = np.sum(P / (1 + mu))
    B = np.sum(P / (mu * (1 + mu)))
    return A, B

def compute_simple_metrics(P, mu, A_val, B_val, A_mat, g):
    # Throughput: φ_i = P_i / (1 + B)
    phi = P / (1 + B_val)
    # Delay D_i
    denom = (1 - A_val) + np.sum(P * (mu / (mu + 1)))
    D = np.zeros_like(P)
    for i in range(P.size):
        num = ((1 - A_val) / mu[i]) + np.sum(
            P * (mu / (mu + 1)**2) + P * (mu / (mu + 1)) / mu[i]
        )
        D[i] = num / denom
    # Update rates Φ_j and AoI_j
    Phi = (phi[:, None] * A_mat).sum(axis=0)
    AoI = np.zeros_like(g)
    for j in range(g.size):
        if Phi[j] > 0:
            weights = (phi * A_mat[:, j]) / Phi[j]
            AoI[j] = g[j] * (1/Phi[j] + np.dot(weights, D))
        else:
            AoI[j] = np.nan
    return phi, D, Phi, AoI

# Example data
np.random.seed(42)
a = b = 100
mask = np.random.rand(a, b) < 0.5
values = np.random.rand(a, b)
A = np.where(mask, 0, values)
mu = np.random.uniform(1.0, 2.0, size=a)
g = np.random.uniform(0.8, 1.2, size=b)

# Original optimization under sum(lambda)=1
def total_aoi_original(lambdas):
    pi_00, pi_i0, pi_ij = compute_steady_state_expanded(lambdas, mu)
    _, _, _, AoI = compute_aoi(A, lambdas, mu, pi_00, pi_i0, pi_ij, g)
    return np.nansum(AoI)

cons_eq = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1)] * a
x0 = np.full(a, 1/a)

res_orig = minimize(total_aoi_original, x0, method='SLSQP', bounds=bounds, constraints=cons_eq)
P_orig = res_orig.x
lambda_orig = P_orig
_, D_orig, Phi_orig, AoI_orig = compute_aoi(A, lambda_orig, mu, *compute_steady_state_expanded(lambda_orig, mu), g)
total_AoI_orig = np.nansum(AoI_orig)

# Simplified optimization
def total_aoi_simple(P):
    A_val, B_val = compute_simple_A_B(P, mu)
    _, _, _, AoI = compute_simple_metrics(P, mu, A_val, B_val, A, g)
    return np.nansum(AoI)

res_simp = minimize(total_aoi_simple, x0, method='SLSQP', bounds=bounds, constraints=cons_eq)
P_simp = res_simp.x
phi_simp, D_simp, Phi_simp, AoI_simp = compute_simple_metrics(P_simp, mu, *compute_simple_A_B(P_simp, mu), A, g)
total_AoI_simp = np.nansum(AoI_simp)

# Prepare comparison tables
df_cam = pd.DataFrame({
    "Orig λ_i": lambda_orig,
    "Simple P_i": P_simp,
    "Orig D_i": D_orig,
    "Simple D_i": D_simp,
}, index=[f"Cam {i+1}" for i in range(a)])

df_obj = pd.DataFrame({
    "g_j": g,
    "Orig Φ_j": Phi_orig,
    "Simple Φ_j": Phi_simp,
    "Orig AoI_j": AoI_orig,
    "Simple AoI_j": AoI_simp,
}, index=[f"Obj {j+1}" for j in range(b)])

df_summary = pd.DataFrame({
    "Method": ["Original", "Simplified"],
    "Total AoI": [total_AoI_orig, total_AoI_simp]
})

# Print to console
print("\nCamera-Level Comparison:")
print(df_cam.to_string(float_format="{:.6f}".format))

print("\nObject-Level Comparison:")
print(df_obj.to_string(float_format="{:.6f}".format))

print("\nSummary:")
print(df_summary.to_string(index=False, float_format="{:.6f}".format))
