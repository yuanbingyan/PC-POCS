import matplotlib.pyplot as plt
import numpy as np

# sparsity = np.array([6, 8, 10, 12, 18, 30])
"""
ssim = np.array([0.892163, 0.864527, 0.861836, 0.833829, 0.772105, 0.664267])
fsim = [0.960006, 0.942181, 0.940555, 0.918241, 0.872195, 0.77338]
psnr = [34.55982, 32.70452, 32.52405, 29.9837, 25.91875, 22.00338]
"""
"""
ssim = np.array([0.824178, 0.787066, 0.776327, 0.740222, 0.699644, 0.610050])
fsim = [0.907083, 0.880687, 0.872954, 0.843707, 0.809582, 0.723280]
psnr = [28.708680, 26.742810, 26.333720, 24.718060, 23.129410, 20.656930]
"""
sparsity = np.array([2, 3, 5, 6, 10])

ssim = np.array([0.865581, 0.836727, 0.774884, 0.747664, 0.696948])
fsim = [0.922602, 0.901536, 0.861761, 0.843561, 0.806302]
psnr = [27.688390, 25.793920, 23.158150, 21.793130, 20.165010]

ssim_threshold = 0.7

# ==== 1. 查找SSIM曲线中跨过0.7的相邻点 ====
for i in range(len(ssim) - 1):
    if (ssim[i] - ssim_threshold) * (ssim[i+1] - ssim_threshold) < 0:
        # 说明这两个点之间跨过0.7
        x0, x1 = sparsity[i], sparsity[i+1]
        y0, y1 = ssim[i], ssim[i+1]
        break

# ==== 2. 做线性插值计算交点位置 ====
slope = (y1 - y0) / (x1 - x0)
x_cross = x0 + (ssim_threshold - y0) / slope
y_cross = ssim_threshold

# ==== 3. 画图 ====
fig, ax1 = plt.subplots(figsize=(6, 4.5))

# 左轴：SSIM、FSIM
line1, = ax1.plot(sparsity, ssim, 'o-', label='SSIM')
line2, = ax1.plot(sparsity, fsim, '^-', label='FSIM')
ax1.axhline(ssim_threshold, color='gray', linestyle='--', linewidth=1)
ax1.text(sparsity[-1]+0.5, ssim_threshold, '', va='center', ha='left', color='gray')
ax1.set_xlabel("Sparsity")
ax1.set_ylabel("SSIM / FSIM")
ax1.grid(True)

# 标出精确交点
ax1.plot(x_cross, y_cross, 'bo')  # 红点
ax1.annotate(f"Sparsity={x_cross:.2f}",
             xy=(x_cross, y_cross),
             xytext=(x_cross + 1.5, y_cross + 0.02),
             arrowprops=dict(arrowstyle="->", color='blue'),
             fontsize=10, color='blue')

# 右轴：PSNR
ax2 = ax1.twinx()
line3, = ax2.plot(sparsity, psnr, 'r*--', label='PSNR')
ax2.set_ylabel("PSNR (dB)")
ax2.tick_params(axis='y')

# 合并图例
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.tight_layout()
plt.show()
