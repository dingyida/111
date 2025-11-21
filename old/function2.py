import sympy as sp

# 1. Define symbols
a, b, u1, u2, C, r = sp.symbols('a b u1 u2 C r', positive=True)
D = a + b + u2

# 2. Compute the un‐normalized CTMC “weights” R for each of the 5 states
R0a = 1
R0b = b*(b + u2) / (u2 * D)
R1  = a*(D + b) / (u1 * D)
R2a = a**2*(D + b) / (u1 * (b + u1) * D)
R2b = a*b / (u2 * D)
N   = R0a + R0b + R1 + R2a + R2b

# 3. Stationary probabilities π
pi0a = R0a / N
pi0b = R0b / N
pi1  = R1  / N
pi2a = R2a / N
pi2b = R2b / N

# 4. Define the unknown 0th‐age moments for each state
v0a0, v0b0, v2b0, v2a0, v1 = sp.symbols('v0a0 v0b0 v2b0 v2a0 v1')

# 5. Write the five balance equations (Theorem 4 style)
eqs = [
    sp.Eq((a + b)*v0a0,             pi0a + u2*v0b0),
    sp.Eq((a + b + u2)*v0b0,        pi0b + b*v0a0 + b*v2b0),
    sp.Eq((b + u2)*v2b0,            pi2b + a*v0b0),
    sp.Eq((b + u1)*v2a0,            pi2a + a*v1),
    sp.Eq((a + u1)*v1,
          pi1 + a*v0a0 + b*v2a0
             + u1*(pi2a/(b + u1))
             + u2*(pi2b/(b + u2)))
]

# 6. Solve for the v‐variables
vars = [v0a0, v0b0, v2b0, v2a0, v1]
A, Cmat = sp.linear_eq_to_matrix(eqs, vars)
sol = A.LUsolve(Cmat)
v0a0_expr, v0b0_expr, v2b0_expr, v2a0_expr, v1_expr = [sp.simplify(s) for s in sol]

# 7. Total AoI F(a,b)
F = v0a0_expr + v0b0_expr + v2b0_expr + v2a0_expr + v1_expr

# 8. Reparametrize under λ1 + λ2 = C via r
a_new = C*r
b_new = C*(1 - r)
G = sp.simplify(F.subs({a: a_new, b: b_new}))

# 9. Find interior critical points
dG_dr       = sp.diff(G, r)
critical_rs = sp.solve(dG_dr, r)
d2G_dr2     = sp.diff(G, r, 2)

# 10. Keep only real minima in (0,1)
feasible_rs = [
    rs for rs in critical_rs
    if rs.is_real and 0 < rs < 1 and sp.simplify(d2G_dr2.subs(r, rs)) > 0
]

# 11. If there is an interior minimum, use it; otherwise use a symbolic Piecewise at the boundaries
if feasible_rs:
    r_star = feasible_rs[0]
else:
    G0 = G.subs(r, 0)
    G1 = G.subs(r, 1)
    r_star = sp.Piecewise(
        (0, G0 < G1),
        (1, True)
    )

# 12. Recover optimal arrival rates and minimum AoI
lambda1_opt = sp.simplify(C * r_star)
lambda2_opt = sp.simplify(C * (1 - r_star))
min_AoI      = sp.simplify(G.subs(r, r_star))

# 13. Display results
print("Optimal r*           =", r_star)
print("Optimal λ1_new       =", lambda1_opt)
print("Optimal λ2_new       =", lambda2_opt)
print("Minimum AoI          =", min_AoI)
