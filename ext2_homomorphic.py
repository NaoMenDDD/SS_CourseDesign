'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-14 09:16:51
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-14 16:41:32
Description: 扩展任务二：同态滤波光照校正 - 频域增强（自适应截止频率）

Copyright (c) 2026 by NaoMenDDD, All Rights Reserved. 
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter

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
    """加载图像，转换为灰度图（0-255范围，float32）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像，请检查格式（支持 .bmp .jpg .png）: {image_path}")
    return img.astype(np.float32)


def compute_fft_spectrum(img):
    """
    计算二维FFT，返回：
        fft_shifted: 频移后的复数频谱
        magnitude_log: 对数幅度谱（用于可视化）
        magnitude_linear: 线性幅度谱（用于能量分析）
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_linear = np.abs(fshift)
    magnitude_log = np.log1p(magnitude_linear)
    return fshift, magnitude_log, magnitude_linear


def compute_cutoff_frequency(fft_shifted, energy_percent=0.90):
    """
    根据径向能量累计比例计算截止频率（半径，单位：像素）
    参数：
        fft_shifted: 频移后的复数频谱
        energy_percent: 累计能量占比阈值，默认0.90（90%）
    返回：
        cutoff_radius: 截止频率对应的半径（浮点数）
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


def visualize_log_spectrum(magnitude_log):
    """将对数幅度谱归一化到0-255范围用于显示"""
    mag_norm = (magnitude_log - np.min(magnitude_log)) / (np.max(magnitude_log) - np.min(magnitude_log) + 1e-8)
    return (mag_norm * 255).astype(np.uint8)


def homomorphic_filter(shape, gamma_l=0.8, gamma_h=2.0, cutoff=40, c=1.0):
    """
    构造同态滤波器（高斯高通型）
    H(u,v) = (gamma_h - gamma_l) * (1 - exp(-c * (D^2 / D0^2))) + gamma_l
    参数：
        shape: (rows, cols)
        gamma_l: 低频增益（<1 压缩光照）
        gamma_h: 高频增益（>1 增强反射细节）
        cutoff: 截止频率 D0（像素半径）
        c: 控制过渡陡峭度，通常1~2
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D**2 / (cutoff**2)))) + gamma_l
    return H


def magnitude_normalize(img):
    """将图像归一化到0-255范围（uint8）"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 1e-8:
        norm = (img - img_min) / (img_max - img_min) * 255.0
    else:
        norm = np.zeros_like(img)
    return norm.astype(np.uint8)


def estimate_illumination_and_reflection(log_img, H_filter):
    """
    根据同态滤波结果分离光照和反射分量（近似）
    用于可视化，返回光照估计和反射估计（线性域）
    """
    fft_log = np.fft.fft2(log_img)
    fft_log_shift = np.fft.fftshift(fft_log)
    filtered_fft = fft_log_shift * H_filter
    f_ishift = np.fft.ifftshift(filtered_fft)
    filtered_log = np.real(np.fft.ifft2(f_ishift))
    # 近似光照分量：对滤波结果做高斯低通
    illu_estimate = gaussian_filter(filtered_log, sigma=15)
    refl_estimate = filtered_log - illu_estimate
    return illu_estimate, refl_estimate


def main(input_image_path, output_dir="output", gamma_l=0.8, gamma_h=2.0, c=1.0, show_output=False):
    """
    主处理流程：
    1. 加载灰度图像
    2. 对数变换（ln）
    3. FFT，自适应计算截止频率 D0
    4. 构造同态滤波器
    5. 应用滤波，逆变换，指数变换
    6. 亮度校正（保持原图均值）
    7. 输出结果并生成组合对比图
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- 1. 加载图像 -----
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)
    rows, cols = img.shape

    # ----- 2. 自适应计算截止频率（基于原图频谱）-----
    fft_shifted_orig, _, _ = compute_fft_spectrum(img)
    D0 = compute_cutoff_frequency(fft_shifted_orig, energy_percent=0.85)
    print(f"自适应截止频率 D0 = {D0:.1f} px")

    # ----- 3. 预处理：对数变换 -----
    eps = 1e-5
    img_safe = img + eps
    log_img = np.log(img_safe)

    # ----- 4. FFT 和对数频谱可视化 -----
    fft_shifted, log_mag, _ = compute_fft_spectrum(log_img)
    spectrum_viz = visualize_log_spectrum(log_mag)

    # ----- 5. 构造同态滤波器 -----
    H_filter = homomorphic_filter(img.shape, gamma_l, gamma_h, D0, c)

    # 可视化滤波器响应（对数压缩）
    filter_response_log = np.log1p(H_filter)
    filter_viz = visualize_log_spectrum(filter_response_log)

    # ----- 6. 应用滤波器并重建 -----
    filtered_fft = fft_shifted * H_filter
    f_ishift = np.fft.ifftshift(filtered_fft)
    filtered_log = np.real(np.fft.ifft2(f_ishift))
    result_img = np.exp(filtered_log) - eps

    # ========== 恢复整体亮度 ==========
    mean_orig = np.mean(img_safe)
    mean_result = np.mean(result_img)
    if mean_result > 1e-6:
        result_img = result_img * (mean_orig / mean_result)
    result_norm = magnitude_normalize(result_img)

    # ----- 7. 近似估计光照和反射分量 -----
    illu_est, refl_est = estimate_illumination_and_reflection(log_img, H_filter)
    illu_viz = magnitude_normalize(np.exp(illu_est))
    refl_viz = magnitude_normalize(np.exp(refl_est))

    # ----- 8. 生成组合结果图 -----
    was_interactive = plt.isinteractive()
    plt.ioff()
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.12,
                          left=0.10, right=0.90, top=0.90, bottom=0.10)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_filter = fig.add_subplot(gs[0, 2])
    ax_illu = fig.add_subplot(gs[1, 0])
    ax_refl = fig.add_subplot(gs[1, 1])
    ax_result = fig.add_subplot(gs[1, 2])

    img_disp = magnitude_normalize(img)
    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=13, fontweight='medium')
    ax_orig.axis('off')

    ax_spec.imshow(spectrum_viz, cmap='gray')
    ax_spec.set_title("Log Spectrum (after ln)", fontsize=13, fontweight='medium')
    ax_spec.axis('off')

    ax_filter.imshow(filter_viz, cmap='hot')
    ax_filter.set_title("Homomorphic Filter Response", fontsize=13, fontweight='medium')
    ax_filter.axis('off')

    ax_illu.imshow(illu_viz, cmap='gray')
    ax_illu.set_title("Estimated Illumination\n(low-frequency)", fontsize=13, fontweight='medium')
    ax_illu.axis('off')

    ax_refl.imshow(refl_viz, cmap='gray')
    ax_refl.set_title("Estimated Reflectance\n(high-frequency)", fontsize=13, fontweight='medium')
    ax_refl.axis('off')

    ax_result.imshow(result_norm, cmap='gray')
    ax_result.set_title("Homomorphic Filtered Image", fontsize=13, fontweight='medium')
    ax_result.axis('off')

    param_text = f"γ_L={gamma_l:.2f}  γ_H={gamma_h:.1f}  D₀={D0:.1f} px  c={c:.1f}\n(adaptive D₀ based on 85% energy)"
    fig.suptitle("Homomorphic Filtering for Illumination Correction", fontsize=20, fontweight='semibold', y=0.96)
    fig.text(0.5, 0.05, param_text, fontsize=11, ha='center', color='#1c1c1e',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#f2f2f6', edgecolor='none'))
    fig.text(0.5, 0.02, "Illumination compressed (γ_L<1) | Reflectance enhanced (γ_H>1)", fontsize=10, ha='center', color='#8e8e93')

    output_path = os.path.join(output_dir, "homomorphic_filtering_result.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.28, facecolor='white', dpi=200)
    plt.close(fig)
    if was_interactive:
        plt.ion()
    if show_output:
        saved_img = cv2.imread(output_path, cv2.IMREAD_COLOR)
        if saved_img is not None:
            saved_img_rgb = cv2.cvtColor(saved_img, cv2.COLOR_BGR2RGB)
            plt.figure(facecolor='white')
            plt.imshow(saved_img_rgb, interpolation='nearest')
            plt.axis('off')
            plt.show()
            plt.close()
        else:
            print(f"⚠️ 无法读取输出图像进行显示: {output_path}")

    print(f"✅ 同态滤波结果已保存至: {output_path}")
    return result_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扩展任务二：同态滤波光照校正 - 频域增强（自适应截止频率）")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径（支持 .bmp .jpg .png），默认 img/house.bmp")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录，默认为 output")
    parser.add_argument("--gamma_l", type=float, default=0.8,
                        help="低频增益（<1 压缩光照），默认0.8")
    parser.add_argument("--gamma_h", type=float, default=2.0,
                        help="高频增益（>1 增强细节），默认2.0")
    parser.add_argument("--c", type=float, default=1.0,
                        help="滤波器过渡陡峭度，默认1.0")
    parser.add_argument("--show", action="store_true",
                        help="显示生成的输出图片")
    args = parser.parse_args()

    # 检查默认路径
    if not os.path.exists(args.input):
        img_folder = Path("img")
        if img_folder.exists() and img_folder.is_dir():
            images_found = list(img_folder.glob("*.bmp")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
            if images_found:
                args.input = str(images_found[0])
                print(f"默认图像不存在，自动选择: {args.input}")
            else:
                raise FileNotFoundError(f"图像文件不存在: {args.input}，且 img 文件夹中无图像。")
        else:
            raise FileNotFoundError(f"图像文件不存在: {args.input}，请检查路径。")

    main(args.input, args.output_dir, args.gamma_l, args.gamma_h, args.c, args.show)
    print("\n处理完成！输出文件：")
    print(f" - {args.output_dir}/homomorphic_filtering_result.png")