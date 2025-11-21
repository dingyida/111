# -*- coding: utf-8 -*-
# Plot decision cycle vs. Average AoI (no inset).
# Curves: Proportional, Proposed (simulated), Proposed (theoretical)

import re
import matplotlib.pyplot as plt

# ==== Input & output ====
log_path = r"C:\Users\dyd\Desktop\picture\decision_cycle.txt"
out_path = "scheme.pdf"

# ==== Parse log file ====
pat = re.compile(
    r"\[Frame\s+(?P<frame>\d+)\]\s+Empirical AoI:\s+"
    r"inc=(?P<inc>[\d\.]+),\s+prop=(?P<prop>[\d\.]+),\s+full=(?P<full>[\d\.]+);\s+"
    r"Theo-inc=(?P<theo>[\d\.]+)"
)

records = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        m = pat.search(line)
        if m:
            records.append((
                int(m.group("frame")),
                float(m.group("prop")),
                float(m.group("full")),
                float(m.group("theo")),
            ))

if not records:
    raise ValueError("No matching lines found. Check the log format or regex.")

# Sort & unpack
records.sort(key=lambda t: t[0])
frames = [t[0] for t in records]
prop   = [t[1] for t in records]
full   = [t[2] for t in records]
theo   = [t[3] for t in records]

# ==== Keep only x in [0, 200] ====
X_MIN, X_MAX = 0, 200
keep = [i for i, x in enumerate(frames) if X_MIN <= x <= X_MAX]
frames = [frames[i] for i in keep]
prop   = [prop[i]   for i in keep]
full   = [full[i]   for i in keep]
theo   = [theo[i]   for i in keep]

# ==== Plot ====
fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_zorder(0)

# Sparser markers: target count -> step
desired_markers = 10
N = max(1, round(len(frames) / desired_markers))

# Lines (Proposed 的两条更粗，标记略大)
ax.plot(frames, prop, marker="s", markevery=N, linewidth=1.6,
        label="Proportional", zorder=6, clip_on=False)
ax.plot(frames, full, marker="o", markevery=N, linewidth=2.2, markersize=6.5,
        label="Proposed (simulated)", zorder=7, clip_on=False)
ax.plot(frames, theo, marker="D", markevery=N, linewidth=2.2, markersize=6.5,
        label="Proposed (theoretical)", zorder=8, clip_on=False)

# Labels / grid / limits
ax.set_xlabel("Decision epoch", fontsize=20)
ax.set_ylabel("Average AoI", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=18)
ax.grid(True, linestyle="--", linewidth=0.6)
ax.set_xlim(X_MIN, X_MAX)
ax.margins(x=0)

# Legend
leg = ax.legend(
    loc="upper left",
    bbox_to_anchor=(0, 0.77, 0.33, 0.22),
    bbox_transform=ax.transAxes,
    frameon=True, ncol=1,
    prop={'size': 13, 'weight': 'bold'},
    borderaxespad=0.3, labelspacing=0.4,
    handlelength=1.6, handletextpad=0.6
)
if leg:
    leg.set_title(None)

# ==== Export ====
fig.savefig(out_path, pad_inches=0.02)
plt.close(fig)
print(f"Saved to: {out_path}")
