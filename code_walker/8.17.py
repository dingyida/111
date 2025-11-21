# Manual-coordinate version per your request:
# - No loops: every text and arrow position is specified directly.
# - Comments are added wherever a text position is defined so you can tweak coordinates quickly.

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(13, 7))
ax.axis("off")

# --- Layout constants
cam_x, cam_w, cam_h = 0.06, 0.14, 0.08
edge_x, edge_y, edge_w, edge_h = 0.40, 0.22, 0.22, 0.56
edge_left, edge_right = edge_x, edge_x + edge_w
ent_x, ent_w, ent_h = 0.78, 0.14, 0.08

# --- Draw Edge Server
ax.add_patch(Rectangle((edge_x, edge_y), edge_w, edge_h, fill=False))
# TEXT POSITION: edge server label (center of the box)
ax.text(edge_x + edge_w/2, edge_y + edge_h/2, "Edge Server",
        ha="center", va="center", fontsize=11)

# Helper to draw an elbow arrow with a single vertical join x (no loops)
def elbow_arrow(start, via_x, end):
    path = Path([start, (via_x, start[1]), (via_x, end[1]), end], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='->', mutation_scale=10, lw=1.5))

# --- Cameras (explicit coordinates, no loops)
# Camera Y positions (top → bottom)
c1_y, c2_y, c3_y, c4_y = 0.82, 0.6067, 0.3933, 0.18
# Edge left ports (clamped inside the box)
pL1_y, pL2_y, pL3_y, pL4_y = 0.74, 0.60, 0.42, 0.26
# Unique join columns (to avoid overlap)
jL1_x, jL2_x, jL3_x, jL4_x = 0.249, 0.283, 0.317, 0.351

# Draw camera boxes
ax.add_patch(Rectangle((cam_x, c1_y - cam_h/2), cam_w, cam_h, fill=False))
ax.add_patch(Rectangle((cam_x, c2_y - cam_h/2), cam_w, cam_h, fill=False))
ax.add_patch(Rectangle((cam_x, c3_y - cam_h/2), cam_w, cam_h, fill=False))
ax.add_patch(Rectangle((cam_x, c4_y - cam_h/2), cam_w, cam_h, fill=False))

# TEXT POSITION: camera labels (centered on each camera box)
ax.text(cam_x + cam_w/2, c1_y, "Camera C1", ha="center", va="center", fontsize=10)
ax.text(cam_x + cam_w/2, c2_y, "Camera C2", ha="center", va="center", fontsize=10)
ax.text(cam_x + cam_w/2, c3_y, "Camera C3", ha="center", va="center", fontsize=10)
ax.text(cam_x + cam_w/2, c4_y, "Camera C4", ha="center", va="center", fontsize=10)

# Elbow arrows from cameras to edge
elbow_arrow((cam_x + cam_w, c1_y), jL1_x, (edge_left, pL1_y))
elbow_arrow((cam_x + cam_w, c2_y), jL2_x, (edge_left, pL2_y))
elbow_arrow((cam_x + cam_w, c3_y), jL3_x, (edge_left, pL3_y))
elbow_arrow((cam_x + cam_w, c4_y), jL4_x, (edge_left, pL4_y))

# TEXT POSITION: λ labels (nudged right to avoid overlap)
ax.text(0.27, c1_y + 0.03, r"Poisson rate $\lambda_1$", ha="center", va="bottom", fontsize=9)
ax.text(0.27, c2_y + 0.03, r"Poisson rate $\lambda_2$", ha="center", va="bottom", fontsize=9)
ax.text(0.27, c3_y + 0.03, r"Poisson rate $\lambda_3$", ha="center", va="bottom", fontsize=9)
ax.text(0.27, c4_y + 0.03, r"Poisson rate $\lambda_4$", ha="center", va="bottom", fontsize=9)

# --- Entities (explicit coordinates, no loops)
# Entity Y positions (top → bottom)
e1_y, e2_y, e3_y, e4_y, e5_y = 0.82, 0.66, 0.50, 0.34, 0.18
# Edge right ports corresponding to entities
pR1_y, pR2_y, pR3_y, pR4_y, pR5_y = 0.75, 0.63, 0.51, 0.39, 0.27
# Unique join columns for right side
jR1_x, jR2_x, jR3_x, jR4_x, jR5_x = 0.6567, 0.6567, 0.6567, 0.6567, 0.6567

# Draw entity boxes
ax.add_patch(Rectangle((ent_x, e1_y - ent_h/2), ent_w, ent_h, fill=False))
ax.add_patch(Rectangle((ent_x, e2_y - ent_h/2), ent_w, ent_h, fill=False))
ax.add_patch(Rectangle((ent_x, e3_y - ent_h/2), ent_w, ent_h, fill=False))
ax.add_patch(Rectangle((ent_x, e4_y - ent_h/2), ent_w, ent_h, fill=False))
ax.add_patch(Rectangle((ent_x, e5_y - ent_h/2), ent_w, ent_h, fill=False))

# TEXT POSITION: entity titles (centered on each entity box)
ax.text(ent_x + ent_w/2, e1_y + 0.015, "Entity E1", ha="center", va="center", fontsize=10)
ax.text(ent_x + ent_w/2, e2_y + 0.015, "Entity E2", ha="center", va="center", fontsize=10)
ax.text(ent_x + ent_w/2, e3_y + 0.015, "Entity E3", ha="center", va="center", fontsize=10)
ax.text(ent_x + ent_w/2, e4_y + 0.015, "Entity E4", ha="center", va="center", fontsize=10)
ax.text(ent_x + ent_w/2, e5_y + 0.015, "Entity E5", ha="center", va="center", fontsize=10)

# TEXT POSITION: per-entity queue captions (shifted down a bit)
ax.text(ent_x + ent_w/2, e1_y - 0.055, "Queue E1: LCFS (direct replade)", ha="center", va="center", fontsize=8)
ax.text(ent_x + ent_w/2, e2_y - 0.055, "Queue E2: LCFS (direct replade)", ha="center", va="center", fontsize=8)
ax.text(ent_x + ent_w/2, e3_y - 0.055, "Queue E3: LCFS (direct replade)", ha="center", va="center", fontsize=8)
ax.text(ent_x + ent_w/2, e4_y - 0.055, "Queue E4: LCFS (direct replade)", ha="center", va="center", fontsize=8)
ax.text(ent_x + ent_w/2, e5_y - 0.055, "Queue E5: LCFS (direct replade)", ha="center", va="center", fontsize=8)

# Elbow arrows from edge to entities
elbow_arrow((edge_right, pR1_y), jR1_x, (ent_x, e1_y))
elbow_arrow((edge_right, pR2_y), jR2_x, (ent_x, e2_y))
elbow_arrow((edge_right, pR3_y), jR3_x, (ent_x, e3_y))
elbow_arrow((edge_right, pR4_y), jR4_x, (ent_x, e4_y))
elbow_arrow((edge_right, pR5_y), jR5_x, (ent_x, e5_y))

# TEXT POSITION: "rates μ_ij → Ej" labels (placed right of the vertical elbow, with vertical staggering and white background)
ax.text(jR1_x + 0.020, pR1_y + 0.040, r"rates $\mu_{ij}$ → E1", ha="left", va="center", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(jR2_x + 0.020, pR2_y + 0.00, r"rates $\mu_{ij}$ → E2", ha="left", va="center", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(jR3_x + 0.020, pR3_y + 0.010, r"rates $\mu_{ij}$ → E3", ha="left", va="center", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(jR4_x + 0.020, pR4_y - 0.020, r"rates $\mu_{ij}$ → E4", ha="left", va="center", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(jR5_x + 0.020, pR5_y - 0.050, r"rates $\mu_{ij}$ → E5", ha="left", va="center", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# --- μ_ij matrix (farther right so it can't overlap anything)
table_str = (
    "    E1     E2     E3     E4     E5\n"
    "C1  μ11    μ12    μ13    μ14    μ15\n"
    "C2  μ21    μ22    μ23    μ24    μ25\n"
    "C3  μ31    μ32    μ33    μ34    μ35\n"
    "C4  μ41    μ42    μ43    μ44    μ45"
)
# TEXT POSITION: matrix top-left anchor (far right margin at x=1.10, centered vertically)
ax.text(1.10, 0.50, r"$\mu_{ij}$ matrix (camera i → entity j):" + "\n" + table_str,
        transform=ax.transAxes, ha="left", va="center", fontsize=9, family="monospace")

plt.tight_layout()
out_path = "system_diagram_manual_coords.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

out_path
