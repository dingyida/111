import numpy as np
import cvxpy as cp

# ————————————————
# 1) Problem data
np.random.seed(42)
a = b = 300
mask   = np.random.rand(a, b) < 0.5
values = np.random.rand(a, b)
A      = np.where(mask, 0.0, values)   # a×b float matrix
mu     = np.random.uniform(1.0, 2.0, size=a)  # service rates (length a)
g      = np.random.uniform(0.8, 1.2, size=b)  # degradation rates (length b)
Λ_max  = 10.0

# ————————————————
# 2) CVXPY variable
λ = cp.Variable(a, nonneg=True)   # λ_i ≥ 0
Λ = cp.sum(λ)                     # total arrival rate

# ————————————————
# 3) CTMC steady-state sums
A_val = cp.sum(cp.multiply(λ, 1.0/(Λ + mu)))            # ∑ λ_i/(Λ+μ_i)
B_val = cp.sum(cp.multiply(λ, 1.0/(mu*(Λ + mu))))       # ∑ λ_i/[μ_i(Λ+μ_i)]

# ————————————————
# 4) Lumped steady-state probabilities
π00        = (1 - A_val) / (1 + Λ * B_val)
π_i0       = cp.multiply(π00, cp.multiply(λ, 1.0/((Λ + mu)*(1 - A_val))))
π_i1_agg   = cp.multiply(cp.multiply(π00, Λ),
                         cp.multiply(λ, 1.0/(mu*(Λ + mu)*(1 - A_val))))
served_prob = π_i0 + π_i1_agg

# ————————————————
# 5) Precompute sums for D
term_vec = mu / (mu + Λ)                             # vector of μ_i/(μ_i+Λ)
S2       = cp.sum(cp.multiply(served_prob, term_vec))              # ∑_i served_i * term_vec_i
S1       = cp.sum(cp.multiply(served_prob * term_vec, 1.0/(mu + Λ)))  # ∑_i served_i*term_vec_i/(μ_i+Λ)
denom    = π00 + S2                                            # π00 + ∑ served_i*(μ_i/(μ_i+Λ))

# ————————————————
# 6) Vectorized conditional delays D (length a)
D = ( (π00 + S2) * (1.0/mu) + S1 ) / denom                     # elementwise

# ————————————————
# 7) Departure rates φ_i
φ = cp.multiply(mu, served_prob)

# ————————————————
# 8) Per-object update rates Φ_j and weighted delays
Φ        = A.T @ φ                    # length b
weighted = A.T @ cp.multiply(φ, D)    # length b

# ————————————————
# 9) Final AoI_j vectorized
AoI = cp.multiply(g, (1.0 + weighted) / Φ)

# ————————————————
# 10) Define & solve the convex problem
objective   = cp.Minimize(cp.sum(AoI))
constraints = [Λ <= Λ_max]
prob        = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=True)

# ————————————————
# 11) Extract & display results
opt_λ    = λ.value
opt_AoI  = AoI.value
opt_total = prob.value

print(f"Minimized total AoI: {opt_total:.6f}")

