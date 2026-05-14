'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-13 14:52:43
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-13 21:48:07
Description: 任务1：频域滤波

Copyright (c) 2026 by NaoMenDDD, All Rights Reserved. 
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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
    """加载图像，转换为灰度图（0-255范围，uint8）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像，请检查格式（支持 .bmp .jpg .png）: {image_path}")
    return img.astype(np.float32)  # 使用浮点数进行处理


def compute_fft_spectrum(img):
    """
    计算二维FFT，并返回：
    - fft_shifted: 频移后的复数频谱
    - magnitude_spectrum: 对数幅度谱（用于可视化）
    - magnitude_linear: 线性幅度谱（用于能量分析）
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_linear = np.abs(fshift)
    magnitude_log = np.log1p(magnitude_linear)
    return fshift, magnitude_log, magnitude_linear


def compute_cutoff_frequency(fft_shifted, energy_percent=0.95):
    """
    根据径向能量累计比例计算截止频率（半径，单位：像素）
    参数：
        fft_shifted: 频移后的复数频谱
        energy_percent: 累计能量占比阈值，默认0.95 (95%)
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


def gaussian_lowpass_filter(shape, D0):
    """
    生成高斯低通滤波器 H_uv
    H(u,v) = exp(-D(u,v)^2 / (2 * D0^2))
    # 标题居中
    shape: (rows, cols)
    D0: 截止频率半径（标准差σ = D0）
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols)
    v = np.arange(rows)
    U, V = np.meshgrid(u, v)
    D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)
    H_lp = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
    return H_lp


def gaussian_highpass_filter(shape, D0):
    """生成高斯高通滤波器: H_hp = 1 - H_lp"""
    H_lp = gaussian_lowpass_filter(shape, D0)
    return 1 - H_lp


def apply_filter_and_reconstruct(fft_shifted, filter_h):
    """
    应用频域滤波器并进行逆变换重建空域图像
    返回：重建图像（0-255范围，uint8）
    """
    filtered_fft = fft_shifted * filter_h
    f_ishift = np.fft.ifftshift(filtered_fft)
    img_reconstructed = np.fft.ifft2(f_ishift)
    img_reconstructed = np.real(img_reconstructed)

    img_min = np.min(img_reconstructed)
    img_max = np.max(img_reconstructed)
    if img_max - img_min > 1e-8:
        img_norm = (img_reconstructed - img_min) / (img_max - img_min) * 255.0
    else:
        img_norm = np.zeros_like(img_reconstructed)
    return img_norm.astype(np.uint8)


def visualize_log_spectrum(magnitude_log):
    """将对数幅度谱归一化到0-255范围用于显示"""
    mag_norm = (magnitude_log - np.min(magnitude_log)) / (np.max(magnitude_log) - np.min(magnitude_log) + 1e-8)
    return (mag_norm * 255).astype(np.uint8)


def main(input_image_path, output_dir="output", show_output=False):
    """
    主处理流程：
    1. 加载灰度图像
    2. 计算FFT及频谱
    3. 基于能量累计占比确定截止频率 D0
    4. 生成高斯低通和高通滤波器
    5. 应用滤波并重建图像
    6. 生成优化布局的组合图（原图+频谱居中，高通在上，低通在下）
    7. 保存最终PNG图像，并标注截止频率及滤波器参数
    8. 按需显示输出结果
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- 1. 加载图像 -----
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)
    rows, cols = img.shape

    # ----- 2. FFT 与频谱 -----
    fft_shifted, magnitude_log, magnitude_linear = compute_fft_spectrum(img)
    magnitude_viz = visualize_log_spectrum(magnitude_log)

    # ----- 3. 确定截止频率（基于能量累计95%）-----
    cutoff_radius = compute_cutoff_frequency(fft_shifted, energy_percent=0.95)
    D0 = cutoff_radius
    print(f"计算得到的截止频率 D0 = {D0:.2f} 像素半径")

    # ----- 4. 生成高斯滤波器 -----
    H_lp = gaussian_lowpass_filter(img.shape, D0)
    H_hp = gaussian_highpass_filter(img.shape, D0)

    # ----- 5. 应用滤波器并重建图像 -----
    fft_lp = fft_shifted * H_lp
    magnitude_lp_log = np.log1p(np.abs(fft_lp))
    img_lp = apply_filter_and_reconstruct(fft_shifted, H_lp)

    fft_hp = fft_shifted * H_hp
    magnitude_hp_log = np.log1p(np.abs(fft_hp))
    img_hp = apply_filter_and_reconstruct(fft_shifted, H_hp)

    magnitude_lp_viz = visualize_log_spectrum(magnitude_lp_log)
    magnitude_hp_viz = visualize_log_spectrum(magnitude_hp_log)

    # ----- 6. 生成流程图式布局 -----
    # 临时关闭交互绘图，避免原始 fig 在保存阶段弹窗
    was_interactive = plt.isinteractive()
    plt.ioff()
    fig = plt.figure(figsize=(26, 13.0), facecolor='white')
    # 调整网格参数使整体视觉舒适
    gs = fig.add_gridspec(3, 5, hspace=0.20, wspace=0.12,
                          width_ratios=[0.55, 1.05, 1.15, 1.05, 1.05],
                          left=0.03, right=0.985, top=0.88, bottom=0.24)

    ax_original = fig.add_subplot(gs[1, 1])       # Original Image
    ax_original_spec = fig.add_subplot(gs[1, 2])  # Original Spectrum (log)

    ax_hp_spec = fig.add_subplot(gs[0, 3])        # High-Pass Filtered Spectrum
    ax_hp_img = fig.add_subplot(gs[0, 4])         # High-Pass Filtered Image

    ax_lp_spec = fig.add_subplot(gs[2, 3])        # Low-Pass Filtered Spectrum
    ax_lp_img = fig.add_subplot(gs[2, 4])         # Low-Pass Filtered Image

    ax_hp_img.text(0.5, -0.04, "High-pass filtering suppresses smooth regions\nand keeps fine details and edges.",
                   transform=ax_hp_img.transAxes, ha='center', va='top', fontsize=9,
                   color='#5c5c5f', wrap=True, clip_on=False)
    ax_lp_img.text(0.5, -0.04, "Low-pass filtering removes high-frequency detail\nand keeps the image smoother.",
                   transform=ax_lp_img.transAxes, ha='center', va='top', fontsize=9,
                   color='#5c5c5f', wrap=True, clip_on=False)

    # 将所有图像转为 uint8 以便 imshow
    images = [img, magnitude_viz, magnitude_lp_viz, img_lp, magnitude_hp_viz, img_hp]
    titles = [
        "Original Image", "Original Spectrum (log)",
        "Low-Pass Filtered Spectrum", "Low-Pass Filtered Image",
        "High-Pass Filtered Spectrum", "High-Pass Filtered Image"
    ]

    for idx, im in enumerate(images):
        if im.dtype != np.uint8:
            if idx == 0:
                im_disp = ((im - im.min()) / (im.max() - im.min() + 1e-8) * 255).astype(np.uint8)
                images[idx] = im_disp
            else:
                if np.max(im) > 0:
                    images[idx] = ((im - im.min()) / (im.max() - im.min() + 1e-8) * 255).astype(np.uint8)
                else:
                    images[idx] = im.astype(np.uint8)

    axes_data = [
        (ax_original, images[0], titles[0]),
        (ax_original_spec, images[1], titles[1]),
        (ax_lp_spec, images[2], titles[2]),
        (ax_lp_img, images[3], titles[3]),
        (ax_hp_spec, images[4], titles[4]),
        (ax_hp_img, images[5], titles[5]),
    ]

    for ax, image_data, title in axes_data:
        ax.imshow(image_data, cmap='gray')
        ax.set_title(title, fontsize=13, fontweight='medium', pad=8)
        ax.axis('off')

    fig.suptitle("Frequency Domain Filtering Pipeline", fontsize=24, fontweight='semibold', 
                 x=0.4, y=0.94, color='#1c1c1e')
    fig.canvas.draw()

    # ------------------- 辅助函数：获取轴边缘坐标（相对 figure 坐标） -------------------
    def _right_center(ax):
        bbox = ax.get_position()
        return bbox.x1, bbox.y0 + bbox.height / 2

    def _left_center(ax):
        bbox = ax.get_position()
        return bbox.x0, bbox.y0 + bbox.height / 2

    def _add_arrow(start, end):
        arrow = FancyArrowPatch(
            start, end,
            transform=fig.transFigure,
            arrowstyle='->',
            color='#1c1c1e',
            lw=1.8,
            mutation_scale=14,
            shrinkA=6,
            shrinkB=6,
        )
        fig.add_artist(arrow)

    # 原始图像 -> 原始频谱
    _add_arrow(_right_center(ax_original), _left_center(ax_original_spec))

    start_point = _right_center(ax_original_spec)

    hp_target = _left_center(ax_hp_spec)
    _add_arrow(start_point, hp_target)

    mid_hp = ((start_point[0] + hp_target[0]) / 2, (start_point[1] + hp_target[1]) / 2)

    fig.text(mid_hp[0] - 0.015, mid_hp[1], f"Gaussian HPF",
             fontsize=9, ha='left', va='center',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.8))

    lp_target = _left_center(ax_lp_spec)
    _add_arrow(start_point, lp_target)
    mid_lp = ((start_point[0] + lp_target[0]) / 2, (start_point[1] + lp_target[1]) / 2)

    fig.text(mid_lp[0] - 0.015, mid_lp[1], f"Gaussian LPF",
             fontsize=9, ha='left', va='center',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.8))

    _add_arrow(_right_center(ax_hp_spec), _left_center(ax_hp_img))

    _add_arrow(_right_center(ax_lp_spec), _left_center(ax_lp_img))

    # 标注截止频率（放置在原始频谱下方）
    cutoff_text = f"Cutoff Frequency D₀ = {D0:.1f} px\n(95% energy accumulation)"
    ax_original_spec.text(
        0.48, -0.30, cutoff_text,
        transform=ax_original_spec.transAxes,
        fontsize=12, ha='center', va='bottom', color='#1c1c1e',
        bbox=dict(boxstyle="round,pad=0.35", facecolor='#f2f2f6', edgecolor='#d1d1d6', alpha=1.0),
        zorder=10,
    )

    # 保存组合图像
    output_path = os.path.join(output_dir, "frequency_filtering_result.png")
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close(fig)
    if was_interactive:
        plt.ion()
    if show_output:
        # 直接显示保存后的图片
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
    print(f"✅ 主结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="灰度图像频域处理：高斯低通/高通滤波")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径（支持 .bmp .jpg .png），默认 img/house.bmp")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录，默认为 output")
    parser.add_argument("--show", action="store_true",
                        help="显示生成的输出图片")
    args = parser.parse_args()

    # 检查默认路径是否存在
    if not os.path.exists(args.input):
        img_folder = Path("img")
        if img_folder.exists() and img_folder.is_dir():
            images_found = list(img_folder.glob("*.bmp")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
            if images_found:
                args.input = str(images_found[0])
                print(f"默认图像不存在，自动选择: {args.input}")
            else:
                raise FileNotFoundError(f"图像文件不存在: {args.input} ，且 img 文件夹中未找到任何 .bmp/.jpg/.png 图像。")
        else:
            raise FileNotFoundError(f"图像文件不存在: {args.input} ，请检查路径或创建 img 文件夹并放入图像。")

    main(args.input, args.output_dir, show_output=args.show)
    print("\n处理完成！输出文件列表：")
    print(f" - {args.output_dir}/frequency_filtering_result.png")