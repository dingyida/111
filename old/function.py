import sympy as sp

# 1. Predefined numeric constants
a_val, b_val = 2.0, 1.0   # arrival rates λ₁, λ₂
u1_val, u2_val = 2.0, 1.0 # service rates μ₁, μ₂

# 2. Compute D and the un-normalized weights R for each state
D_val = a_val + b_val + u2_val

R0a = 1
R0b = b_val * (b_val + u2_val) / (u2_val * D_val)
R1  = a_val * (D_val + b_val) / (u1_val * D_val)
R2a = a_val**2 * (D_val + b_val) / (u1_val * (b_val + u1_val) * D_val)
R2b = a_val * b_val / (u2_val * D_val)

# 3. Normalize to get π
N_val  = R0a + R0b + R1 + R2a + R2b
pi0a   = R0a / N_val
pi0b   = R0b / N_val
pi1    = R1  / N_val
pi2a   = R2a / N_val
pi2b   = R2b / N_val

# 4. Display the stationary distribution
print(f"π₀ₐ = {pi0a:.6f}")
print(f"π₀ᵦ = {pi0b:.6f}")
print(f"π₁  = {pi1:.6f}")
print(f"π₂ₐ = {pi2a:.6f}")
print(f"π₂ᵦ = {pi2b:.6f}")
