# Plot Average AoI vs. maximum uploading rate (WITHOUT inset).
# X 轴改成对数刻度；其余保持不变。
import matplotlib.pyplot as plt

# ---------------- Data (Lambda_max, as requested) ----------------
data = [
    {"Lambda_max": 40.0, "slsqp": 4.6910, "Proportional": 6.5809, "proposed": 4.1277, "Theo-proposed": 4.1469},
    {"Lambda_max": 20.0, "slsqp": 5.3488, "Proportional": 6.5948, "proposed": 4.1574, "Theo-proposed": 4.1789},
    {"Lambda_max": 10.0, "slsqp": 4.2135, "Proportional": 5.5620, "proposed": 4.2133, "Theo-proposed": 4.2760},
    {"Lambda_max":  5.0, "slsqp": 4.3357, "Proportional": 5.7038, "proposed": 4.3159, "Theo-proposed": 4.4383},
    {"Lambda_max":  2.5, "slsqp": 4.6225, "Proportional": 6.0573, "proposed": 4.6139, "Theo-proposed": 4.8856},
]

# Sort by Lambda_max ascending
data_sorted = sorted(data, key=lambda d: d["Lambda_max"])

x           = [d["Lambda_max"] for d in data_sorted]
y_slsqp     = [d["slsqp"] for d in data_sorted]
y_prop      = [d["Proportional"] for d in data_sorted]
y_proposed  = [d["proposed"] for d in data_sorted]
y_th_prop   = [d["Theo-proposed"] for d in data_sorted]

# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)

# === NEW: X 轴设为对数刻度（指数轴）。base=2 使 5→10→20→40 间距更均匀 ===
ax.set_xscale('log', base=2)

# Put spines under the data; ensure points/lines cover axes
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_zorder(0)

# Proportional first (behind)
ax.plot(
    x, y_prop, marker="s", markersize=6, linewidth=1.6,
    label="Proportional", zorder=6, clip_on=False
)

# Three close series
close_series = {
    "slsqp": y_slsqp,
    "proposed": y_proposed,
    "Theo-proposed": y_th_prop,
}
label_map = {
    "slsqp": "SLSQP",
    "proposed": "Proposed (simulated)",
    "Theo-proposed": "Proposed (theoretical)"
}

avg_y = {k: sum(v)/len(v) for k, v in close_series.items()}
order = sorted(avg_y.keys(), key=lambda k: avg_y[k])

marker_map = {"slsqp": "o", "proposed": "^", "Theo-proposed": "D"}
sizes = [11, 8, 5]
lws   = [2.4, 2.0, 1.6]
base_z = 7

for j, name in enumerate(order):
    ax.plot(
        x, close_series[name],
        marker=marker_map[name], markersize=sizes[j],
        linewidth=lws[j], label=label_map[name],
        zorder=base_z + j, clip_on=False
    )

# Labels & ticks
ax.set_xlabel("Maximum uploading rate", fontsize=20)
ax.set_ylabel("Average AoI", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=18)

# Limit & ticks (在对数轴上固定到你的数据点，并显示为原值)
ax.set_xlim(min(x), max(x))
ax.margins(x=0)
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in x])  # 显示 2.5, 5, 10, 20, 40

# 网格：主/次刻度都显示
ax.grid(True, which="both", linestyle="--", linewidth=0.6)

# Legend positioned & sized by coordinates
leg = ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.10, 0.80, 0.34, 0.20),  # (x, y, w, h) in axes coords
    bbox_transform=ax.transAxes,
    frameon=True, ncol=1,
    prop={'size': 13, 'weight': 'bold'},
    borderaxespad=0.3, labelspacing=0.4,
    handlelength=1.6, handletextpad=0.6
)
if leg:
    leg.set_title(None)

# ---------------- Export ----------------
pdf_path = "compare2.pdf"
fig.savefig(pdf_path,pad_inches=0.02)
plt.close(fig)

print(pdf_path)
