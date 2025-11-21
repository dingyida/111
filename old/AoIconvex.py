import numpy as np
from scipy.optimize import minimize

# 1) Problem data (same A, B, P, μ as before)
a, b = 5, 8
np.random.seed(2)
p = 0.5
while True:
    A = (np.random.rand(a, b) < p).astype(float)
    if (A.sum(axis=0) > 0).all():
        break
B = np.random.rand(b)
P, mu, eps = 1.0, 1.2, 1e-6

# 2) Precompute constants C1, C2 (since ∑λ = P ⇒ ρ = P/μ is fixed)
rho = P/mu
num = (1 + rho + rho**2)**2 + 2*rho**3
den = (1 + rho + rho**2)*(1 + rho)**2 + eps
C1 = num/den
C2 = 1 + rho**2/(1 + rho)

# 3) Define the convex objective: ∑ᵢ B[i]*(1/μ)*(C1 + C2*(μ/( (Aᵀλ)[i] + eps )))
def objective(lam):
    rate = A.T.dot(lam) + eps        # rate_i = ∑ₙ A_{n,i} λₙ
    aoi_terms = B*(1/mu)*(C1 + C2*(mu/rate))
    return np.sum(aoi_terms)

# 4) Constraints and bounds
#    - ∑ λₙ = P  (equality)
#    - λₙ ≥ 0    (bounds)
constraints = ({'type': 'eq', 'fun': lambda lam: lam.sum() - P},)
bounds      = [(0, None)] * a

# 5) Solve with SLSQP
lam0 = np.ones(a)*(P/a)  # uniform start
res = minimize(objective, lam0, method='SLSQP',
               bounds=bounds, constraints=constraints)

print("Optimal λ:", res.x)
print("Sum λ   :", res.x.sum())
print("AoI loss:", res.fun)
