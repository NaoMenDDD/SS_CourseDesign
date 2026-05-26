'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-14 16:36:28
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-26 17:14:41
Description: 扩展任务C：频域高通滤波与 Sobel 边缘检测对比（自适应截止频率）

Copyright (c) 2026 by NaoMenDDD, All Rights Reserved. 
'''


import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# 样式设置（与项目保持一致）
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 200,
    'figure.dpi': 120,
    'image.cmap': 'gray',
    'text.color': '#1c1c1e',
    'axes.labelcolor': '#1c1c1e',
    'xtick.color': '#8e8e93',
    'ytick.color': '#8e8e93'
})


def load_grayscale_image(image_path):
    """
    加载图像并转换为灰度图
    参数：
        image_path: 图像文件路径（支持 .bmp .jpg .png）
    返回：
        灰度图数组（0-255范围，float32）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img.astype(np.float32)


def compute_image_entropy(img_uint8):
    """计算图像的灰度熵，反映纹理复杂度"""
    hist = cv2.calcHist([img_uint8], [0], None, [256], [0, 256])
    hist = hist.flatten() / np.sum(hist)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def compute_spectral_slope(fft_shifted):
    """
    计算频谱能量径向衰减斜率（对数域）
    返回斜率值（绝对值越大表示能量衰减越快，图像越平滑）
    """
    magnitude_sq = np.abs(fft_shifted) ** 2
    rows, cols = magnitude_sq.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_r = int(np.ceil(np.max(dist)))
    radial_energy = np.zeros(max_r + 1)
    for r in range(max_r + 1):
        mask = (dist >= r) & (dist < r + 1)
        radial_energy[r] = np.sum(magnitude_sq[mask])
    # 归一化能量
    total = np.sum(radial_energy)
    if total > 0:
        radial_energy /= total
    # 取中段半径（避免直流和极高频噪声）
    r_start = 5
    r_end = min(80, max_r - 5)
    if r_end <= r_start:
        return 0.5
    r_vals = np.arange(r_start, r_end)
    y_vals = radial_energy[r_start:r_end]
    # 避免 log(0)
    y_vals = np.maximum(y_vals, 1e-6)
    # 对数域线性拟合
    coeffs = np.polyfit(np.log(r_vals + 1), np.log(y_vals), 1)
    slope = coeffs[0]  # 负值，绝对值越大衰减越快
    return abs(slope)


def compute_cutoff_frequency_adaptive(fft_shifted, img_uint8, filter_type='highpass'):
    """
    基于频谱斜率和图像熵的自适应截止频率计算
    参数：
        fft_shifted: 频移后的复数频谱
        img_uint8: 原始灰度图（uint8）
        filter_type: 滤波类型，'lowpass' 或 'highpass'
    返回：
        cutoff_radius: 截止频率半径（浮点数）
    """
    magnitude_sq = np.abs(fft_shifted) ** 2
    rows, cols = magnitude_sq.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_r = int(np.ceil(np.max(dist)))
    radial_energy = np.zeros(max_r + 1)
    for r in range(max_r + 1):
        mask = (dist >= r) & (dist < r + 1)
        radial_energy[r] = np.sum(magnitude_sq[mask])
    cum_ratio = np.cumsum(radial_energy) / np.sum(radial_energy)
    
    # 计算频谱斜率
    slope = compute_spectral_slope(fft_shifted)
    # 计算图像熵
    entropy = compute_image_entropy(img_uint8)
    
    # 动态能量百分比公式（经验调参）
    # 平滑图像（斜率大、熵小）→ 百分比偏低（约0.85）
    # 纹理丰富图像（斜率小、熵大）→ 百分比偏高（约0.95）
    percent = 0.85 + 0.1 * np.tanh(slope - 0.5) + 0.05 * np.tanh((entropy - 7) / 2)
    percent = np.clip(percent, 0.80, 0.97)
    if filter_type == 'highpass':
        percent += 0.03
        percent = np.clip(percent, 0.80, 0.97)
    
    cutoff_idx = np.where(cum_ratio >= percent)[0]
    cutoff_radius = cutoff_idx[0] if len(cutoff_idx) > 0 else max_r
    return float(cutoff_radius)


def compute_cutoff_frequency_energy(fft_shifted, energy_percent=0.95):
    """
    根据径向能量累计比例计算截止频率（半径，单位：像素）
    参数：
        fft_shifted: 频移后的复数频谱
        energy_percent: 累计能量占比阈值，默认0.95 (95%)
    返回：
        cutoff_radius: 截止频率对应的半径（浮点数）
    说明：此函数为传统方法，保留用于比较。
    """
    magnitude_sq = np.abs(fft_shifted) ** 2
    rows, cols = magnitude_sq.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_radius = int(np.ceil(np.max(dist)))
    radial_energy = np.zeros(max_radius + 1)
    for r in range(max_radius + 1):
        mask = (dist >= r) & (dist < (r + 1))
        radial_energy[r] = np.sum(magnitude_sq[mask])
    cum_energy = np.cumsum(radial_energy)
    total_energy = cum_energy[-1]
    cum_ratio = cum_energy / total_energy
    cutoff_idx = np.where(cum_ratio >= energy_percent)[0]
    cutoff_radius = cutoff_idx[0] if len(cutoff_idx) > 0 else max_radius
    return float(cutoff_radius)


def ideal_highpass_filter(shape, D0):
    """
    构造理想高通滤波器
    参数：
        shape: 图像尺寸 (rows, cols)
        D0: 截止频率（像素半径）
    返回：
        滤波器频率响应矩阵（0-1）
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    H = np.ones(shape, dtype=np.float32)
    H[D <= D0] = 0.0
    return H


def gaussian_highpass_filter(shape, D0):
    """
    构造高斯高通滤波器
    参数：
        shape: 图像尺寸 (rows, cols)
        D0: 截止频率（像素半径）
    返回：
        滤波器频率响应矩阵（0-1）
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D2 = U**2 + V**2
    H_lp = np.exp(-D2 / (2 * (D0 ** 2)))
    return 1 - H_lp


def apply_filter_and_reconstruct(fft_shifted, filter_h):
    """
    应用频域滤波器并进行反变换重建图像
    参数：
        fft_shifted: 频移后的复数频谱
        filter_h: 滤波器频率响应
    返回：
        重建后的图像（uint8，范围0-255）
    """
    filtered = fft_shifted * filter_h
    img_recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_recon = np.real(img_recon)
    img_recon = np.abs(img_recon)
    img_min, img_max = img_recon.min(), img_recon.max()
    if img_max - img_min > 1e-8:
        img_recon = (img_recon - img_min) / (img_max - img_min) * 255.0
    else:
        img_recon = np.zeros_like(img_recon)
    return img_recon.astype(np.uint8)


def sobel_edge_detection(img_uint8):
    """
    使用 Sobel 算子进行边缘检测
    参数：
        img_uint8: 输入灰度图（uint8）
    返回：
        梯度幅度图（uint8）
    """
    grad_x = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0
    return mag_norm.astype(np.uint8)


def normalize_display(img):
    """将任意图像归一化到 0-255 范围用于显示"""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(img).astype(np.uint8)


def main(input_image_path, output_dir="output", cutoff_method="adaptive", show_output=False):
    """
    主处理流程：比较频域高通滤波与 Sobel 边缘检测
    参数：
        input_image_path: 输入图像路径
        output_dir: 输出目录
        cutoff_method: 截止频率计算方法，'adaptive' 或 'energy'
        show_output: 是否显示输出图片
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- 1. 加载图像 -----
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)
    img_uint8 = img.astype(np.uint8)

    # ----- 2. 计算自适应截止频率（用于高通滤波）-----
    fft_orig = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_orig)
    
    if cutoff_method == "adaptive":
        D0 = compute_cutoff_frequency_adaptive(fft_shifted, img_uint8, filter_type='highpass')
        print(f"自适应方法（基于频谱斜率和图像熵）计算的截止频率 D0 = {D0:.1f} px")
    else:
        D0 = compute_cutoff_frequency_energy(fft_shifted, energy_percent=0.95)
        print(f"能量累计95%方法计算的截止频率 D0 = {D0:.1f} px")

    # ----- 3. 同时构造并应用两种高通滤波器 -----
    H_hp_ideal = ideal_highpass_filter(img.shape, D0)
    H_hp_gaussian = gaussian_highpass_filter(img.shape, D0)
    img_hp_ideal = apply_filter_and_reconstruct(fft_shifted, H_hp_ideal)
    img_hp_gaussian = apply_filter_and_reconstruct(fft_shifted, H_hp_gaussian)

    # ----- 4. Sobel 边缘检测 -----
    img_sobel = sobel_edge_detection(img_uint8)

    # ----- 5. 生成对比结果图（2x2布局：原图/理想高通/高斯高通/Sobel）-----
    fig = plt.figure(figsize=(10, 9), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.05,
                          left=0.05, right=0.95, top=0.90, bottom=0.19)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_sobel = fig.add_subplot(gs[0, 1])
    ax_hp_ideal = fig.add_subplot(gs[1, 0])
    ax_hp_gaussian = fig.add_subplot(gs[1, 1])

    # 显示原图
    img_disp = normalize_display(img)
    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=12, fontweight='medium')
    ax_orig.axis('off')

    # Sobel 结果
    ax_sobel.imshow(img_sobel, cmap='gray')
    ax_sobel.set_title("Sobel (Gradient Magnitude)", fontsize=11, fontweight='medium')
    ax_sobel.axis('off')

    # 理想高通结果
    ax_hp_ideal.imshow(img_hp_ideal, cmap='gray')
    ax_hp_ideal.set_title(f"Frequency Domain Ideal Highpass\nD₀={D0:.1f}px", fontsize=11, fontweight='medium')
    ax_hp_ideal.axis('off')

    # 高斯高通结果
    ax_hp_gaussian.imshow(img_hp_gaussian, cmap='gray')
    ax_hp_gaussian.set_title(f"Frequency Domain Gaussian Highpass\nD₀={D0:.1f}px", fontsize=11, fontweight='medium')
    ax_hp_gaussian.axis('off')

    # 添加底部说明文字（方法对比）
    text_str = (
        "Comparison Summary:\n\n"
        "• Frequency-domain HPF (global enhancement):\n"
        "  - Ideal HPF: sharper transition, stronger details, possible ringing.\n"
        "  - Gaussian HPF: smoother transition, fewer ringing artifacts.\n\n"
        "• Sobel (spatial domain): local gradient approximation, fast and stable,\n"
        "  with relatively thicker edges under strong contrast."
    )
    fig.text(0.5, 0.055, text_str, ha='center', va='center', fontsize=9.8,
             fontfamily='monospace', color='#1c1c1e',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#f2f2f6', edgecolor='none'))

    fig.suptitle("Comparison: Ideal/Gaussian Highpass vs. Sobel Edge Detection",
                 fontsize=14, fontweight='semibold', y=0.965)

    # ----- 6. 保存结果 -----
    output_path = os.path.join(output_dir, "hpf_vs_sobel.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.28, facecolor='white', dpi=200)
    plt.close(fig)

    # ----- 7. 可选显示 -----
    if show_output:
        saved = cv2.imread(output_path, cv2.IMREAD_COLOR)
        if saved is not None:
            saved_rgb = cv2.cvtColor(saved, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(saved_rgb)
            plt.axis('off')
            plt.show()

    print(f"✅ 对比图已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高通滤波 vs Sobel 边缘检测对比（自适应截止频率）")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--cutoff_method", type=str, default="adaptive", choices=["adaptive", "energy"],
                        help="截止频率计算方法：adaptive（自适应，基于频谱斜率和图像熵）或 energy（能量累计95%），默认 adaptive")
    parser.add_argument("--show", action="store_true", help="显示结果图片")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        img_folder = Path("img")
        if img_folder.exists():
            imgs = list(img_folder.glob("*.bmp")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
            if imgs:
                args.input = str(imgs[0])
                print(f"默认图像不存在，自动选择: {args.input}")
            else:
                raise FileNotFoundError(f"未找到图像: {args.input}")
    main(args.input, args.output_dir, args.cutoff_method, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/hpf_vs_sobel.png")