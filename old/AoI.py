import torch
import torch.nn as nn
import torch.optim as optim

# ── Problem dims & constants ─────────────────────────────────────────────
a, b = 5, 8        # number of cameras, number of objects
P    = 1.0         # total upload-rate budget
mu   = 1.2         # server service rate μ
eps  = 1e-6        # numerical safeguard
p    = 0.5         # probability for random 1's in A

torch.manual_seed(2)

# ── Option A: rejection‐sampling so every column has ≥1 coverage ─────────
while True:
    A = (torch.rand(a, b) < p).float()   # each entry =1 with prob = p
    if (A.sum(dim=0) > 0).all():         # check no column is all zeros
        break

# ── Random B ──────────────────────────────────────────────────────────────
B = torch.rand(b)

print("A =\n", A)
print("col-sums =", A.sum(dim=0))   # should all be ≥1
print("B =\n", B)

# ── Model: FC net mapping (A,B) → λ with ∑λ = P via linear scaling ───────
class LambdaNet(nn.Module):
    def __init__(self, a, b, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(a * b + b, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, a)

    def forward(self, A, B):
        x = torch.cat([A.view(-1), B], dim=0)
        x = torch.relu(self.fc1(x))
        z = self.fc2(x)                    # raw scores, shape [a]
        # linear scaling to enforce λ_i ≥ 0 and ∑ λ_i = P
        z_pos = z - z.min()                # shift so min = 0
        z_pos = z_pos + eps                # avoid exact zeros
        λ = P * z_pos / z_pos.sum()        # scale to sum to P
        return λ

model     = LambdaNet(a, b)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── Loss = ∑ᵢ B[i]*(1/μ)*(C1 + C2*(μ/Σₙ λₙ)) per object then summed ───────
def aoi_loss_per_object(A, B, λ, mu):
    # 1) per-camera utilizations and total ρ
    rho_n = λ.div(mu)        # [a]
    rho   = rho_n.sum()      # scalar

    # 2) C1 = α_W(ρ)
    num   = (1 + rho + rho**2)**2 + 2 * rho**3
    den   = (1 + rho + rho**2) * (1 + rho)**2 + eps
    C1    = num.div(den)    # scalar

    # 3) C2 = 1 + ρ² / (1 + ρ)
    C2    = 1 + rho**2 / (1 + rho + eps)

    # 4) rate covering each object i
    rate_per_obj = A.t().mv(λ)  # [b]

    # 5) AoI_i = B[i] * (1/μ) * (C1 + C2 * (μ/(rate_i + eps)))
    aoi_i = B * ((1/mu) * (C1 + C2 * (mu / (rate_per_obj + eps))))

    # 6) sum over all objects
    return aoi_i.sum()

# ── Training loop ────────────────────────────────────────────────────────
for epoch in range(1, 20001):
    optimizer.zero_grad()
    λ    = model(A, B)
    loss = aoi_loss_per_object(A, B, λ, mu)
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:03d}  loss={loss.item():.4f}  λ={λ.detach().numpy()}")

# ── Final result ─────────────────────────────────────────────────────────
print("\nFinal λ:", λ.detach().numpy())
print("sum λ  :", λ.sum().item())  # should be ≈ P
