import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple

def compute_steady_state_expanded(
    lambda_arrivals: np.ndarray,
    mu: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    a = lambda_arrivals.size
    Λ = np.sum(lambda_arrivals)
    A_val = np.sum(lambda_arrivals / (Λ + mu))
    B_val = np.sum(lambda_arrivals / (mu * (Λ + mu)))

    pi_00 = (1.0 - A_val) / (1.0 + Λ * B_val)
    pi_i0 = pi_00 * lambda_arrivals / ((Λ + mu) * (1.0 - A_val))
    pi_i1_agg = pi_00 * (lambda_arrivals * Λ) / (mu * (Λ + mu) * (1.0 - A_val))
    pi_ij = np.outer(pi_i1_agg, lambda_arrivals / Λ)

    return pi_00, pi_i0, pi_ij

def compute_aoi(
    A: np.ndarray,
    lambda_arrivals: np.ndarray,
    mu: np.ndarray,
    pi_00: float,
    pi_i0: np.ndarray,
    pi_ij: np.ndarray,
    g: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b = A.shape
    Λ = np.sum(lambda_arrivals)

    phi = mu * (pi_i0 + pi_ij.sum(axis=1))
    served_probs = pi_i0 + pi_ij.sum(axis=1)
    denom = pi_00 + np.sum(served_probs * (mu / (mu + Λ)))

    D = np.zeros(a)
    for i in range(a):
        term0 = pi_00 / mu[i]
        termk = np.sum(
            served_probs * (mu / (mu + Λ)) * (1/(mu + Λ) + 1/mu[i])
        )
        D[i] = (term0 + termk) / denom

    Phi = (phi[:, None] * A).sum(axis=0)
    AoI = np.zeros(b)
    for j in range(b):
        if Phi[j] > 0:
            weights = (phi * A[:, j]) / Phi[j]
            AoI[j] = g[j] * (1/Phi[j] + np.dot(weights, D))
        else:
            AoI[j] = np.nan

    return phi, D, Phi, AoI

# Example with 10 cameras and 10 objects
np.random.seed(42)
a = b = 10
mask = np.random.rand(a, b) < 0.5                # 50% 为 True（即我们设为 0）
values = np.random.rand(a, b)                    # [0, 1) 的浮点数
A = np.where(mask, 0, values)                    # 如果 mask 为 True，设为 0；否则取随机值

mu = np.random.uniform(1.0, 2.0, size=a)
g = np.random.uniform(0.8, 1.2, size=b)

# Objective: minimize total AoI over lambda_arrivals
def total_aoi(lambdas):
    pi_00, pi_i0, pi_ij = compute_steady_state_expanded(lambdas, mu)
    _, _, _, AoI = compute_aoi(A, lambdas, mu, pi_00, pi_i0, pi_ij, g)
    return np.nansum(AoI)

# Constraint: sum(lambda_i) <= 10
cons = ({'type': 'ineq', 'fun': lambda x: 10 - np.sum(x)})
bounds = [(0, None)] * a
x0 = np.full(a, 10.0 / a)


# Run optimization
result = minimize(total_aoi, x0, method='SLSQP', bounds=bounds, constraints=cons)
opt_lambda = result.x
opt_total_aoi = result.fun

# Compute final metrics
pi_00_opt, pi_i0_opt, pi_ij_opt = compute_steady_state_expanded(opt_lambda, mu)
_, _, Phi_opt, AoI_opt = compute_aoi(A, opt_lambda, mu, pi_00_opt, pi_i0_opt, pi_ij_opt, g)

# Display results
df_lambda = pd.DataFrame(opt_lambda, index=[f"Cam {i+1}" for i in range(a)], columns=["λ_i"])
df_object = pd.DataFrame({
    "g_j": g,
    "Φ_j": Phi_opt,
    "AoI_j": AoI_opt
}, index=[f"Obj {j+1}" for j in range(b)])

print("\nOptimized Camera Rates (λ_i):")
print(df_lambda.to_string(float_format="{:.6f}".format))
print("\nOptimized Object-Level AoI:")
print(df_object.to_string(float_format="{:.6f}".format))
print(f"\nTotal AoI after optimization: {opt_total_aoi:.6f}")
