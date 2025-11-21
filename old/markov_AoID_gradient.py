import numpy as np

# 1. Steady-state computation
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

# 2. AoI computation
def compute_aoi(A: np.ndarray, lambda_arrivals: np.ndarray, mu: np.ndarray,
                pi_00: float, pi_i0: np.ndarray, pi_ij: np.ndarray, g: np.ndarray):
    a, b = A.shape
    Λ = np.sum(lambda_arrivals)
    phi = mu * (pi_i0 + pi_ij.sum(axis=1))
    served = pi_i0 + pi_ij.sum(axis=1)
    denom = pi_00 + np.sum(served * (mu / (mu + Λ)))

    D = np.zeros(a)
    for i in range(a):
        term0 = pi_00 / mu[i]
        termk = np.sum(served * (mu / (mu + Λ)) * (1/(mu + Λ) + 1/mu[i]))
        D[i] = (term0 + termk) / denom

    Phi = (phi[:,None] * A).sum(axis=0)
    AoI = np.zeros(b)
    for j in range(b):
        if Phi[j] > 0:
            w = (phi * A[:,j]) / Phi[j]
            AoI[j] = g[j] * (1/Phi[j] + np.dot(w, D))
        else:
            AoI[j] = np.nan

    return phi, D, Phi, AoI

# 3. Projection onto simplex {x>=0, sum=x0}
def project_simplex(v: np.ndarray, z: float) -> np.ndarray:
    if np.all(v >= 0) and np.isclose(v.sum(), z):
        return v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v)+1) > (sv - z))[0][-1]
    theta = (sv[rho] - z) / (rho + 1)
    return np.maximum(v - theta, 0)

# 4. Gradient descent with projection
def gradient_descent(total_aoi, x0, Lambda_max,
                     lr=0.1, max_iters=2000, tol=1e-6, h=1e-6):
    x = x0.copy()
    for it in range(1, max_iters+1):
        f0 = total_aoi(x)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += h
            grad[i] = (total_aoi(xp) - f0) / h
        x_new = x - lr * grad
        x_new = np.maximum(x_new, 0)
        if x_new.sum() > Lambda_max:
            x_new = project_simplex(x_new, Lambda_max)
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {it} iterations")
            return x_new
        x = x_new
    print("Reached max iterations")
    return x

# 5. Setup problem
np.random.seed(42)
a = b = 300
mask = np.random.rand(a, b) < 0.5                # 50% 为 True（即我们设为 0）
values = np.random.rand(a, b)                    # [0, 1) 的浮点数
A = np.where(mask, 0, values)                    # 如果 mask 为 True，设为 0；否则取随机值
mu = np.random.uniform(1.0, 2.0, size=a)
g  = np.random.uniform(0.8, 1.2, size=b)
Lambda_max = 10.0

def total_aoi_wrapper(lambdas):
    lambdas = np.maximum(lambdas, 0)
    pi_00, pi_i0, pi_ij = compute_steady_state_expanded(lambdas, mu)
    _, _, _, AoI = compute_aoi(A, lambdas, mu, pi_00, pi_i0, pi_ij, g)
    return np.nansum(AoI)

# 6. Run gradient descent
x0 = np.full(a, Lambda_max / a)
opt_lambda = gradient_descent(total_aoi_wrapper, x0, Lambda_max, lr=0.5, max_iters=3000)

# 7. Report
opt_total = total_aoi_wrapper(opt_lambda)
print("Optimal λ_i per camera:")
for i, v in enumerate(opt_lambda, 1):
    print(f" Cam {i}: λ_{i} = {v:.6f}")
print(f"\nMinimized total AoI: {opt_total:.6f}")
