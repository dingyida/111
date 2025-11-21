import os
import cv2
import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────
POOL_KERNEL_SIZE     = 5    # size of the max‐pooling window (3×3 by default)
RGB_CANNY_THRESH1    = 50   # lower threshold for RGB Canny
RGB_CANNY_THRESH2    = 150  # upper threshold for RGB Canny
DEPTH_CANNY_THRESH1  = 20
DEPTH_CANNY_THRESH2  = 150

# ─── Paths ───────────────────────────────────────────────────────────────────
dataset_rgb    = 'dataset3/rgb'
rgb_filename   = 'r-0-1.png'
rgb_path       = os.path.join(dataset_rgb, rgb_filename)

dataset_depth  = 'dataset3/depth'
depth_filename = 'd-0-1.png'
depth_path     = os.path.join(dataset_depth, depth_filename)

# ─── Load images ─────────────────────────────────────────────────────────────
img_rgb   = cv2.imread(rgb_path)
if img_rgb is None:
    raise FileNotFoundError(f"Cannot find RGB image: {rgb_path}")

img_depth = cv2.imread(depth_path)
if img_depth is None:
    raise FileNotFoundError(f"Cannot find depth image: {depth_path}")

# ─── 1. Compute per-channel Canny on RGB ─────────────────────────────────────
b, g, r    = cv2.split(img_rgb)
edges_r    = cv2.Canny(r, RGB_CANNY_THRESH1, RGB_CANNY_THRESH2)
edges_g    = cv2.Canny(g, RGB_CANNY_THRESH1, RGB_CANNY_THRESH2)
edges_b    = cv2.Canny(b, RGB_CANNY_THRESH1, RGB_CANNY_THRESH2)

# ─── 2. Merge & pool RGB edges, but keep edges_b unpooled ──────────────────
# Create 3×3 pooling kernel
kernel = np.ones((POOL_KERNEL_SIZE, POOL_KERNEL_SIZE), dtype=np.uint8)

# First merge R and G, then dilate (max-pool) that merge
merged_rg = cv2.bitwise_or(edges_r, edges_g)
pooled_rg = cv2.dilate(merged_rg, kernel)

# Now merge the pooled RG result with the original (unpooled) B
merged_rgb = cv2.bitwise_or(pooled_rg, edges_b)

# ─── 3. Compute depth-map-2 via 3×3 max-pooling of depth Canny edges ───────
depth_gray = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
edges_d    = cv2.Canny(depth_gray, DEPTH_CANNY_THRESH1, DEPTH_CANNY_THRESH2)
depth_map2 = cv2.dilate(edges_d, kernel)

# ─── 4. Merge rule (unchanged): intersection of RGB merge & depth_map2 ────
merged_and = cv2.bitwise_and(merged_rgb, depth_map2)

# ─── 5. Display & save results ──────────────────────────────────────────────
cv2.imshow('Merged RG (pooled) + B (unpooled)', merged_rgb)
cv2.imshow(f'Depth map 2 (pooled {POOL_KERNEL_SIZE}×{POOL_KERNEL_SIZE})', depth_map2)
cv2.imshow('Intersection (RGB & Depth)', merged_and)

cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = 'intersection_edges.png'
cv2.imwrite(output_path, merged_and)
print(f"Saved final intersection mask to: {output_path}")
