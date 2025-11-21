import numpy as np

# Numeric constants
u1_val, u2_val = 2, 1.5  # service rates μ₁, μ₂
C_val = 2.0               # total arrival‐rate budget

def compute_AoI(r, C_val, u1_val, u2_val):
    """
    Compute the total AoI for a given split r,
    where λ1 = C_val * r, λ2 = C_val * (1-r).
    Solves the 5-state CTMC age-moment equations numerically.
    """
    a = C_val * r
    b = C_val * (1 - r)
    D = a + b + u2_val

    # Un-normalized CTMC weights R_i
    R0a = 1.0
    R0b = b*(b + u2_val)/(u2_val * D)
    R1  = a*(D + b)/(u1_val * D)
    R2a = a**2*(D + b)/(u1_val*(b + u1_val)*D)
    R2b = a*b/(u2_val * D)

    N = R0a + R0b + R1 + R2a + R2b
    pi0a, pi0b, pi1 = R0a/N, R0b/N, R1/N
    pi2a, pi2b = R2a/N, R2b/N

    # Build and solve the linear system A v = Cvec
    A = np.zeros((5, 5))
    Cvec = np.zeros(5)

    # Eq1: (a+b)*v0a0 - u2*v0b0 = pi0a
    A[0, 0], A[0, 1], Cvec[0] = a + b, -u2_val, pi0a

    # Eq2: -b*v0a0 + (a+b+u2)*v0b0 - b*v2b0 = pi0b
    A[1, 0], A[1, 1], A[1, 2], Cvec[1] = -b, a + b + u2_val, -b, pi0b

    # Eq3: -a*v0b0 + (b+u2)*v2b0 = pi2b
    A[2, 1], A[2, 2], Cvec[2] = -a, b + u2_val, pi2b

    # Eq4: (b+u1)*v2a0 - a*v1 = pi2a
    A[3, 3], A[3, 4], Cvec[3] = b + u1_val, -a, pi2a

    # Eq5: -a*v0a0 - b*v2a0 + (a+u1)*v1 = pi1 + u1*(pi2a/(b+u1)) + u2*(pi2b/(b+u2))
    A[4, 0], A[4, 3], A[4, 4] = -a, -b, a + u1_val
    Cvec[4] = pi1 + u1_val*(pi2a/(b + u1_val)) + u2_val*(pi2b/(b + u2_val))

    v = np.linalg.solve(A, Cvec)
    return v.sum()

# Gradient descent parameters
lr = 0.1      # learning rate
iterations = 2000
r = 0.5       # initial guess

for i in range(iterations):
    eps = 1e-6
    grad = (compute_AoI(r + eps, C_val, u1_val, u2_val) -
            compute_AoI(r - eps, C_val, u1_val, u2_val)) / (2 * eps)
    r -= lr * grad
    r = min(max(r, 0.0), 1.0)  # project back into [0,1]

# Final solution
r_star = r
lambda1_opt = C_val * r_star
lambda2_opt = C_val * (1 - r_star)
min_AoI = compute_AoI(r_star, C_val, u1_val, u2_val)

print(f"Optimal split r*     = {r_star}")
print(f"Optimal λ1_new       = {lambda1_opt}")
print(f"Optimal λ2_new       = {lambda2_opt}")
print(f"Minimum AoI          = {min_AoI}")
