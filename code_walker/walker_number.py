# Plot AoI vs. walker_number WITHOUT inset.
# Narrower figure width; legend positioned/sized by coordinates; enlarged axis labels and tick labels.
import matplotlib.pyplot as plt

# ---------------- Data (as requested) ----------------
data = [
    {"walker": 3,  "slsqp": 5.1409, "Proportional": 5.5470, "proposed": 5.1396, "Theo-proposed": 5.1500},
    {"walker": 5,  "slsqp": 3.8839, "Proportional": 4.4792, "proposed": 3.8818, "Theo-proposed": 3.8777},
    {"walker": 7,  "slsqp": 4.6910, "Proportional": 6.5809, "proposed": 4.1277, "Theo-proposed": 4.1469},
    {"walker": 9,  "slsqp": 6.9167, "Proportional": 5.4786, "proposed": 4.1771, "Theo-proposed": 4.1858},
    {"walker": 11, "slsqp": 4.0685, "Proportional": 6.4869, "proposed": 3.9253, "Theo-proposed": 3.9328},
]

# Sort by walker_number ascending
data_sorted = sorted(data, key=lambda d: d["walker"])

x           = [d["walker"] for d in data_sorted]
y_slsqp     = [d["slsqp"] for d in data_sorted]
y_prop      = [d["Proportional"] for d in data_sorted]
y_proposed  = [d["proposed"] for d in data_sorted]
y_th_prop   = [d["Theo-proposed"] for d in data_sorted]

# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)

# 让网格/刻度在下，脊线 zorder 更低；点与线盖住坐标轴
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_zorder(0)

# Proportional
ax.plot(
    x, y_prop, marker="s", markersize=6, linewidth=1.6,
    label="Proportional", zorder=6, clip_on=False
)

# 三条近似曲线
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

# 轴标签与刻度
ax.set_xlabel("The number of AR users", fontsize=20)
ax.set_ylabel("Average AoI", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=18)

# 去掉 x 轴两端空白
ax.set_xlim(min(x), max(x))
ax.margins(x=0)
ax.set_xticks(x)
ax.set_ylim(top=7.8)
ax.grid(True, linestyle="--", linewidth=0.6)

# 图例用坐标定位与尺寸
leg = ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.0, 0.78, 0.34, 0.20),  # (x, y, w, h) in ax coords
    bbox_transform=ax.transAxes,
    frameon=True, ncol=1,
    prop={'size': 13, 'weight': 'bold'},
    borderaxespad=0.3, labelspacing=0.4,
    handlelength=1.6, handletextpad=0.6
)
if leg:
    leg.set_title(None)

# ---------------- Export ----------------
pdf_path = "compare1.pdf"
fig.savefig(pdf_path, pad_inches=0.02)
plt.close(fig)

print(pdf_path)
