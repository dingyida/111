import os
import cv2

# Paths
dataset_dir = 'dataset3/rgb'
rgb_name    = 'r-0-1.png'
rgb_path    = os.path.join(dataset_dir, rgb_name)

dataset_dir2 = 'dataset3/depth'
depth_name   = 'd-0-1.png'
depth_path   = os.path.join(dataset_dir2, depth_name)

# Load RGB and depth
img_rgb = cv2.imread(rgb_path)
if img_rgb is None:
    raise FileNotFoundError(f"无法找到 RGB 图像: {rgb_path}")

img_depth = cv2.imread(depth_path)
if img_depth is None:
    raise FileNotFoundError(f"无法找到深度图像: {depth_path}")

# Split RGB channels and Canny
b, g, r = cv2.split(img_rgb)
edges_r = cv2.Canny(r, 50, 150)
edges_g = cv2.Canny(g, 50, 150)
edges_b = cv2.Canny(b, 50, 150)

merged_rgb_edges = cv2.bitwise_or(edges_r, edges_g)
merged_rgb_edges = cv2.bitwise_or(merged_rgb_edges, edges_b)

# Convert depth to grayscale & Canny
depth_gray = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
edges_d    = cv2.Canny(depth_gray, 25, 150)

# Fuse RGB‐edges with depth‐edges
merged_all = cv2.bitwise_or(merged_rgb_edges, edges_d)

# Visualize
cv2.imshow('Merged RGB Edges',     merged_rgb_edges)
cv2.imshow('Depth Edges (edges_d)', edges_d)
cv2.imshow('All Edges Combined',    merged_all)

cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = 'all_edges_combined.png'
cv2.imwrite(output_path, merged_all)