# -*- coding: utf-8 -*-
import re

# 初始化列表
inc_vals = []
prop_vals = []
full_vals = []
inv_vals = []

# 打开并逐行读取
with open(r"C:\Users\dyd\Desktop\picture\11.txt", 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 使用正则提取 inc, prop, full, inv 后面的数值
        m = re.search(r'inc=([0-9.]+),\s*prop=([0-9.]+),\s*full=([0-9.]+),\s*inv=([0-9.]+)', line)
        if m:
            inc_vals.append(float(m.group(1)))
            prop_vals.append(float(m.group(2)))
            full_vals.append(float(m.group(3)))
            inv_vals.append(float(m.group(4)))

# 校验帧数
n = len(inc_vals)
if n != 200:
    print(f"警告：实际读取到 {n} 帧，预期 200 帧。请检查文件格式。")

# 计算平均值
avg_inc  = sum(inc_vals)  / n
avg_prop = sum(prop_vals) / n
avg_full = sum(full_vals) / n
avg_inv  = sum(inv_vals)  / n

# 输出
print(f"平均 inc  = {avg_inc:.4f}")
print(f"平均 prop = {avg_prop:.4f}")
print(f"平均 full = {avg_full:.4f}")
print(f"平均 inv  = {avg_inv:.4f}")
