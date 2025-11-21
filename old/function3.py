import sympy as sp
import numpy as np

# Numeric constants
u1_val, u2_val = 1.0, 1.0  # μ₁, μ₂
C_val = 2.0               # λ₁ + λ₂ budget

# Decision variable
r = sp.symbols('r', real=True)

# Reparametrize λ₁, λ₂ under the budget constraint
a = C_val * r
b = C_val * (1 - r)

# Normalization constant D
D = a + b + u2_val

# Un-normalized CTMC weights
R0a = 1
R0b = b * (b + u2_val) / (u2_val * D)
R1  = a * (D + b) / (u1_val * D)
R2a = a**2 * (D + b) / (u1_val * (b + u1_val) * D)
R2b = a * b / (u2_val * D)

# Stationary probabilities
N    = R0a + R0b + R1 + R2a + R2b
pi0a = R0a / N
pi0b = R0b / N
pi1  = R1  / N
pi2a = R2a / N
pi2b = R2b / N

# Age moments symbols
v0a0, v0b0, v2b0, v2a0, v1 = sp.symbols('v0a0 v0b0 v2b0 v2a0 v1')

# Balance equations
eqs = [
    sp.Eq((a + b)*v0a0,             pi0a + u2_val * v0b0),
    sp.Eq((a + b + u2_val)*v0b0,    pi0b + b * v0a0 + b * v2b0),
    sp.Eq((b + u2_val)*v2b0,        pi2b + a * v0b0),
    sp.Eq((b + u1_val)*v2a0,        pi2a + a * v1),
    sp.Eq((a + u1_val)*v1,
          pi1  + a * v0a0 + b * v2a0
               + u1_val*(pi2a/(b + u1_val))
               + u2_val*(pi2b/(b + u2_val)))
]

# Solve for v-expressions
vars = [v0a0, v0b0, v2b0, v2a0, v1]
A, Cmat = sp.linear_eq_to_matrix(eqs, vars)
v_sol = A.LUsolve(Cmat)
v0a0_expr, v0b0_expr, v2b0_expr, v2a0_expr, v1_expr = v_sol

# Total AoI function G(r)
F = v0a0_expr + v0b0_expr + v2b0_expr + v2a0_expr + v1_expr
G = sp.simplify(F)

# Numeric minimization via sampling
G_func = sp.lambdify(r, G, 'numpy')
rs = np.linspace(0, 1, 10001)
Gs = G_func(rs)
idx_min = int(np.nanargmin(Gs))
r_star = float(rs[idx_min])
min_AoI = float(Gs[idx_min])

# Compute optimal rates
lambda1_opt = C_val * r_star
lambda2_opt = C_val * (1 - r_star)

# Display results
print(f"Optimal split r*     = {r_star}")
print(f"Optimal λ1_new       = {lambda1_opt}")
print(f"Optimal λ2_new       = {lambda2_opt}")
print(f"Minimum AoI          = {min_AoI}")

