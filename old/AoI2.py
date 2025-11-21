import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import time

# ── Problem constants ────────────────────────────────────────────────────────
a, b = 5, 8        # # cameras, # objects
P    = 1.0         # total upload-rate budget
mu   = 1.2         # service rate
eps  = 1e-6        # numeric safeguard

# Precompute C1, C2 (since ρ = P/μ is fixed)
rho = P / mu
num = (1 + rho + rho**2)**2 + 2 * rho**3
den = (1 + rho + rho**2) * (1 + rho)**2 + eps
C1  = num / den
C2  = 1 + rho**2 / (1 + rho)

def aoi_loss_np(A, B, lam):
    """Compute AoI loss in numpy for given A,B,λ."""
    rate = A.T.dot(lam) + eps           # [b]
    return np.sum(B * (1/mu) * (C1 + C2 * (mu / rate)))

def solve_convex(A, B):
    """Convex: minimize over λ with SLSQP and simplex constraint."""
    def obj(lam):
        return aoi_loss_np(A, B, lam)
    lam0 = np.ones(a) * (P/a)
    cons = ({'type':'eq', 'fun': lambda lam: lam.sum() - P},)
    bounds = [(0, None)] * a
    t0 = time.time()
    res = minimize(obj, lam0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x, res.fun, time.time() - t0

class DirectLambda(nn.Module):
    def __init__(self, a, P):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(a))
        self.P = P
    def forward(self):
        return torch.softmax(self.logits, dim=0) * self.P

def solve_direct_nn(A, B, epochs=20000, lr=1e-2):
    """NN: train a tiny softmax net for logits→λ."""
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    model = DirectLambda(a, P)
    opt   = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        lam = model()
        rate = A_t.t().mv(lam) + eps
        loss = (B_t * (1/mu) * (C1 + C2 * (mu / rate))).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        lam_np  = model().cpu().numpy()
        loss_np = loss.item()
    return lam_np, loss_np, time.time() - t0

# ── Run comparisons ─────────────────────────────────────────────────────────
np.random.seed(0)
results = []
N = 5
for i in range(N):
    # 1) sample A s.t. every column has ≥1 coverage
    while True:
        A = (np.random.rand(a, b) < 0.5).astype(float)
        if (A.sum(axis=0) > 0).all():
            break
    B = np.random.rand(b)

    # 2) solve both ways
    lam_c, loss_c, t_c = solve_convex(A, B)
    lam_n, loss_n, t_n = solve_direct_nn(A, B)

    results.append({
        'instance':      i+1,
        'loss_convex':   loss_c,
        'time_convex_s': t_c,
        'loss_direct':   loss_n,
        'time_direct_s': t_n,
    })

# 3) print summary
print(f"{'inst':>4}  {'loss_C':>10}  {'t_C(s)':>8}  {'loss_NN':>10}  {'t_NN(s)':>8}")
for r in results:
    print(f"{r['instance']:4d}  {r['loss_convex']:10.4f}  {r['time_convex_s']:8.3f}  "
          f"{r['loss_direct']:10.4f}  {r['time_direct_s']:8.3f}")
