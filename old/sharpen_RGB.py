import os
import cv2

# 1. 设置路径和文件名
dataset_dir = 'dataset3/rgb'
img_name = 'r-0-1.png'  # 修改为你实际的文件名
img_path = os.path.join(dataset_dir, img_name)

# 2. 读取原图（BGR 格式）
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"无法在 {img_path} 找到图像")

# 3. 将 BGR 拆为三个单通道（注意 OpenCV 默认顺序是 B,G,R）
b_channel, g_channel, r_channel = cv2.split(img)

# 4. 对每个通道应用 Canny
#    阈值可以针对不同通道做微调
edges_r = cv2.Canny(r_channel, threshold1=50, threshold2=150)
edges_g = cv2.Canny(g_channel, threshold1=50, threshold2=150)
edges_b = cv2.Canny(b_channel, threshold1=50, threshold2=150)

# 5. 可视化
cv2.imshow('Original', img)
cv2.imshow('Edges - R channel', edges_r)
cv2.imshow('Edges - G channel', edges_g)
cv2.imshow('Edges - B channel', edges_b)

cv2.waitKey(0)
cv2.destroyAllWindows()
