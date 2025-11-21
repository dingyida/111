import os
import cv2

# 1. 设置 dataset 文件夹路径和图像文件名
dataset_dir = 'dataset3/rgb'
img_name = 'r-0-1.png'  # 修改为你实际的文件名

# 2. 拼接出图像的完整路径并读取（BGR 格式）
img_path = os.path.join(dataset_dir, img_name)
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"无法在 {img_path} 找到图像")

# 3. 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. 应用 Canny 边缘检测
#    这里 threshold1、threshold2 可根据图像对比度调节
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# 5. 可视化结果
cv2.imshow('Original Image', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
