import numpy as np
from typing import Tuple

def compute_steady_state_expanded(
    lambda_arrivals: np.ndarray,
    mu: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the steady-state probabilities for the CTMC, with expanded buffer-source states.

    States:
      (0,0): idle, buffer empty
      (i,0): server busy on camera i, buffer empty
      (i,j): server busy on camera i, buffer holds a frame from camera j

    Returns π_00, π_i0, and full π_ij matrix.
    """
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
    """
    Compute AoI given CTMC steady-state (with expanded π_ij).

    Returns φ_i, D_i, Φ_j, and AoI_j.
    """
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

# Random incidence matrix, ensure each object seen by >=1 camera
A = (np.random.rand(a, b) < 0.5).astype(int)
for j in range(b):
    if A[:, j].sum() == 0:
        A[np.random.randint(0, a), j] = 1

# Random arrival/service/growth rates
lambda_arrivals = np.random.uniform(0.5, 1.5, size=a)
mu = np.random.uniform(1.0, 2.0, size=a)
g = np.random.uniform(0.8, 1.2, size=b)

# Steady-state and AoI
pi_00, pi_i0, pi_ij = compute_steady_state_expanded(lambda_arrivals, mu)
phi, D, Phi, AoI = compute_aoi(A, lambda_arrivals, mu, pi_00, pi_i0, pi_ij, g)

print("AoI for 10 objects:\n", AoI)
