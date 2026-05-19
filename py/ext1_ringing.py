'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-14 16:36:11
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-19 17:09:31
Description: 扩展任务一：对比理想低通、巴特沃斯低通、高斯低通的振铃效应

Copyright (c) 2026 by NaoMenDDD, All Rights Reserved. 
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# 设置matplotlib全局样式
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
    """加载灰度图像，返回 float32 数组"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img.astype(np.float32)


def compute_cutoff_frequency(fft_shifted, energy_percent=0.95):
    """根据径向能量累计比例计算截止频率（像素半径）"""
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
    cum_ratio = cum_energy / cum_energy[-1]
    cutoff_idx = np.where(cum_ratio >= energy_percent)[0]
    return float(cutoff_idx[0]) if len(cutoff_idx) > 0 else float(max_radius)


def ideal_lowpass_filter(shape, D0):
    """理想低通滤波器"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    H = np.zeros(shape, dtype=np.float32)
    H[D <= D0] = 1.0
    return H


def butterworth_lowpass_filter(shape, D0, n=2):
    """巴特沃斯低通滤波器，n为阶数"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    H = 1.0 / (1.0 + (D / (D0 + 1e-8)) ** (2 * n))
    return H


def gaussian_lowpass_filter(shape, D0):
    """高斯低通滤波器"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D2 = U**2 + V**2
    H = np.exp(-D2 / (2 * (D0 ** 2)))
    return H


def apply_filter_and_reconstruct(fft_shifted, filter_h):
    """应用频域滤波器并重建空域图像"""
    filtered = fft_shifted * filter_h
    img_recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_recon = np.real(img_recon)
    # 归一化到 [0,255]
    img_min, img_max = img_recon.min(), img_recon.max()
    if img_max - img_min > 1e-8:
        img_recon = (img_recon - img_min) / (img_max - img_min) * 255.0
    else:
        img_recon = np.zeros_like(img_recon)
    return img_recon.astype(np.uint8)


def normalize_display(img):
    """将任意图像归一化到0-255并转为uint8"""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img_norm = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img_norm = np.zeros_like(img)
    return img_norm.astype(np.uint8)


def main(input_image_path, output_dir="output", butterworth_order=2, show_output=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载图像
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)

    # 计算FFT和自适应截止频率（基于原图能量95%）
    fft_orig = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_orig)
    D0 = compute_cutoff_frequency(fft_shifted, energy_percent=0.95)
    print(f"自适应截止频率 D0 = {D0:.1f} px")

    # 构造滤波器
    H_ideal = ideal_lowpass_filter(img.shape, D0)
    H_butter = butterworth_lowpass_filter(img.shape, D0, n=butterworth_order)
    H_gauss = gaussian_lowpass_filter(img.shape, D0)

    # 应用滤波并重建
    img_ideal = apply_filter_and_reconstruct(fft_shifted, H_ideal)
    img_butter = apply_filter_and_reconstruct(fft_shifted, H_butter)
    img_gauss = apply_filter_and_reconstruct(fft_shifted, H_gauss)

    # 生成组合图（2行2列：原图，理想，巴特沃斯，高斯）
    fig = plt.figure(figsize=(14, 12), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.12,
                          left=0.10, right=0.90, top=0.86, bottom=0.10)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_ideal = fig.add_subplot(gs[0, 1])
    ax_butter = fig.add_subplot(gs[1, 0])
    ax_gauss = fig.add_subplot(gs[1, 1])

    img_disp = normalize_display(img)
    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=13, fontweight='medium')
    ax_orig.axis('off')

    ax_ideal.imshow(img_ideal, cmap='gray')
    ax_ideal.set_title(f"Ideal Lowpass\nD₀={D0:.1f}px (severe ringing)", fontsize=12, fontweight='medium')
    ax_ideal.axis('off')

    ax_butter.imshow(img_butter, cmap='gray')
    ax_butter.set_title(f"Butterworth Lowpass (n={butterworth_order})\nD₀={D0:.1f}px (moderate ringing)", fontsize=12, fontweight='medium')
    ax_butter.axis('off')

    ax_gauss.imshow(img_gauss, cmap='gray')
    ax_gauss.set_title(f"Gaussian Lowpass\nD₀={D0:.1f}px (no ringing)", fontsize=12, fontweight='medium')
    ax_gauss.axis('off')

    fig.suptitle("Comparison of Lowpass Filters: Ringing Effect", fontsize=18, fontweight='semibold', y=0.95)
    fig.text(0.5, 0.03, "Ideal filter causes strong ringing due to sharp cut-off; Butterworth smooths transition; Gaussian has no ringing.",
             fontsize=10, ha='center', color='#8e8e93', style='italic')

    output_path = os.path.join(output_dir, "ringing_comparison.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.28, facecolor='white', dpi=200)
    plt.close(fig)

    if show_output:
        saved = cv2.imread(output_path, cv2.IMREAD_COLOR)
        if saved is not None:
            saved_rgb = cv2.cvtColor(saved, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(saved_rgb)
            plt.axis('off')
            plt.show()

    print(f"✅ 振铃效应对比图已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扩展任务一：对比理想低通、巴特沃斯低通、高斯低通的振铃效应")
    parser.add_argument("--input", "-i", type=str, default="../img/house.bmp",
                        help="输入图像路径")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--order", type=int, default=2,
                        help="巴特沃斯滤波器的阶数，默认2")
    parser.add_argument("--show", action="store_true",
                        help="显示结果图片")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        img_folder = Path("../img")
        if img_folder.exists():
            imgs = list(img_folder.glob("*.bmp")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
            if imgs:
                args.input = str(imgs[0])
                print(f"默认图像不存在，自动选择: {args.input}")
    main(args.input, args.output_dir, args.order, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/ringing_comparison.png")