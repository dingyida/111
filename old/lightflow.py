import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to two consecutive frames
dataset_dir = 'dataset3/rgb'
img1_path = os.path.join(dataset_dir, 'r-0-1.png')
img2_path = os.path.join(dataset_dir, 'r-66676-2.png')

# Read frames
frame1 = cv2.imread(img1_path)
frame2 = cv2.imread(img2_path)
if frame1 is None or frame2 is None:
    raise FileNotFoundError(f"Cannot find one of the images: {img1_path}, {img2_path}")

# Convert to gray
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Compute dense optical flow
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
flow_x, flow_y = flow[...,0], flow[...,1]

# --- your existing visualizations ---
# 1) HSV color map
magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
hsv[...,0] = angle / 2
hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
flow_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('flow_hsv.png', flow_hsv)
cv2.imshow('Optical Flow (HSV)', flow_hsv)

# 2) Quiver arrows
step = 16
h, w = gray1.shape
y, x = np.mgrid[step/2:h:step, step/2:w:step]
fx = flow_x[step//2::step, step//2::step]
fy = flow_y[step//2::step, step//2::step]
plt.figure(figsize=(w/100, h/100), dpi=100)
plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
plt.quiver(x, y, fx, fy, angles='xy', scale_units='xy', scale=1, width=0.002)
plt.axis('off')
plt.tight_layout()
plt.savefig('flow_quiver.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# --- new: compute and display residuals ---
# 3) Compute spatial gradients on frame1
Ix = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
# 4) Temporal gradient
It = gray2 - gray1

# 5) Residual map: |Ix * u + Iy * v + It|
residual = np.abs(Ix * flow_x + Iy * flow_y + It)

# 6) Normalize for visualization
res_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
res_norm = res_norm.astype(np.uint8)

cv2.imwrite('flow_residual.png', res_norm)
cv2.imshow('Flow Residuals', res_norm)

# 7) (Optional) threshold to get motion‐boundary mask
_, motion_mask = cv2.threshold(res_norm, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Motion Boundaries', motion_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
