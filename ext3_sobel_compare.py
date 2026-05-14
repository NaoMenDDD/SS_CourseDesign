'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-14 16:36:28
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-14 16:44:40
Description: 扩展任务三：频域高通滤波与 Sobel 边缘检测对比

Copyright (c) 2026 by NaoMenDDD, All Rights Reserved. 
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# 样式设置
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img.astype(np.float32)


def compute_cutoff_frequency(fft_shifted, energy_percent=0.95):
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


def ideal_highpass_filter(shape, D0):
    """理想高通滤波器：1 - 理想低通"""
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
    """高斯高通滤波器：1 - 高斯低通"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D2 = U**2 + V**2
    H_lp = np.exp(-D2 / (2 * (D0 ** 2)))
    return 1 - H_lp


def apply_filter_and_reconstruct(fft_shifted, filter_h):
    filtered = fft_shifted * filter_h
    img_recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_recon = np.real(img_recon)
    # 归一化到 [0,255]（高通结果可能包含负值，取绝对值并归一化）
    img_recon = np.abs(img_recon)
    img_min, img_max = img_recon.min(), img_recon.max()
    if img_max - img_min > 1e-8:
        img_recon = (img_recon - img_min) / (img_max - img_min) * 255.0
    else:
        img_recon = np.zeros_like(img_recon)
    return img_recon.astype(np.uint8)


def sobel_edge_detection(img_uint8):
    """Sobel边缘检测，返回梯度幅值图（0-255）"""
    grad_x = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0
    return mag_norm.astype(np.uint8)


def normalize_display(img):
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(img).astype(np.uint8)


def main(input_image_path, output_dir="output", filter_type="ideal", show_output=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载图像
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)
    img_uint8 = img.astype(np.uint8)

    # FFT及自适应截止频率
    fft_orig = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_orig)
    D0 = compute_cutoff_frequency(fft_shifted, energy_percent=0.95)
    print(f"自适应截止频率 D0 = {D0:.1f} px (用于高通滤波)")

    # 构造高通滤波器
    if filter_type.lower() == "ideal":
        H_hp = ideal_highpass_filter(img.shape, D0)
        filter_name = "Ideal Highpass"
    else:  # gaussian
        H_hp = gaussian_highpass_filter(img.shape, D0)
        filter_name = "Gaussian Highpass"

    # 频域高通滤波
    img_hp = apply_filter_and_reconstruct(fft_shifted, H_hp)

    # Sobel边缘检测
    img_sobel = sobel_edge_detection(img_uint8)

    # 显示差异图（高通结果与Sobel的差值）
    # 两者范围不同，都需要归一化显示
    diff = np.abs(img_hp.astype(np.float32) - img_sobel.astype(np.float32))
    diff_norm = normalize_display(diff)

    # 生成组合图（2行2列：原图，高通结果，Sobel结果，差异图）
    fig = plt.figure(figsize=(14, 12), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.12,
                          left=0.10, right=0.90, top=0.86, bottom=0.10)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_hp = fig.add_subplot(gs[0, 1])
    ax_sobel = fig.add_subplot(gs[1, 0])
    ax_diff = fig.add_subplot(gs[1, 1])

    img_disp = normalize_display(img)
    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=13, fontweight='medium')
    ax_orig.axis('off')

    ax_hp.imshow(img_hp, cmap='gray')
    ax_hp.set_title(f"Frequency Domain {filter_name}\nD₀={D0:.1f}px", fontsize=12, fontweight='medium')
    ax_hp.axis('off')

    ax_sobel.imshow(img_sobel, cmap='gray')
    ax_sobel.set_title("Sobel Edge Detection (Spatial Domain)", fontsize=12, fontweight='medium')
    ax_sobel.axis('off')

    ax_diff.imshow(diff_norm, cmap='gray')
    ax_diff.set_title("Difference Map\n(|HPF - Sobel|)", fontsize=12, fontweight='medium')
    ax_diff.axis('off')

    fig.suptitle("Comparison: Frequency Domain Highpass vs. Sobel Edge Detection", fontsize=16, fontweight='semibold', y=0.95)
    fig.text(0.5, 0.03, "Highpass filtering enhances edges but may cause ringing; Sobel gives cleaner edges with built-in smoothing.",
             fontsize=10, ha='center', color='#8e8e93', style='italic')

    output_path = os.path.join(output_dir, "hpf_vs_sobel.png")
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

    print(f"✅ 频域高通 vs Sobel 对比图已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扩展任务三：频域高通滤波与 Sobel 边缘检测对比")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--filter_type", type=str, default="ideal", choices=["ideal", "gaussian"],
                        help="高通滤波器类型：ideal 或 gaussian，默认 ideal")
    parser.add_argument("--show", action="store_true", help="显示结果图片")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        img_folder = Path("img")
        if img_folder.exists():
            imgs = list(img_folder.glob("*.bmp")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
            if imgs:
                args.input = str(imgs[0])
                print(f"默认图像不存在，自动选择: {args.input}")
    main(args.input, args.output_dir, args.filter_type, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/hpf_vs_sobel.png")