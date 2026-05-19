'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-19 22:00:00
LastEditors: NaoMenDDD 2017954808@qq.com
Description: 扩展任务三子任务：频域高通滤波 vs Sobel 边缘检测对比

对比两种边缘提取方法：
- 路径A (频域)：原图 → FFT → 高通滤波 → IFFT → 边缘图
- 路径B (空域)：原图 → Sobel 算子 → 梯度幅值

输出组合对比图，包含原图、频域高通结果、Sobel 结果及方法说明。
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


def compute_cutoff_frequency(fft_shifted, energy_percent=0.95):
    """
    根据径向能量累计比例计算自适应截止频率（用于滤波器设计）
    参数：
        fft_shifted: 频移后的复数频谱
        energy_percent: 累计能量占比阈值（默认0.95=95%）
    返回：
        截止频率对应的半径（浮点数，单位：像素）
    """
    magnitude_sq = np.abs(fft_shifted) ** 2
    rows, cols = magnitude_sq.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_radius = int(np.ceil(np.max(dist)))
    radial_energy = np.zeros(max_radius + 1)
    # 计算各个半径处的能量
    for r in range(max_radius + 1):
        mask = (dist >= r) & (dist < (r + 1))
        radial_energy[r] = np.sum(magnitude_sq[mask])
    cum_energy = np.cumsum(radial_energy)
    cum_ratio = cum_energy / cum_energy[-1]
    # 找到能量达到阈值的最小半径
    cutoff_idx = np.where(cum_ratio >= energy_percent)[0]
    return float(cutoff_idx[0]) if len(cutoff_idx) > 0 else float(max_radius)


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
    img_recon = np.abs(img_recon)   # 高通结果可能有负值，取绝对值
    # 归一化到 0-255
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
    # 归一化到 0-255
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0
    return mag_norm.astype(np.uint8)


def normalize_display(img):
    """将任意图像归一化到 0-255 范围用于显示"""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(img).astype(np.uint8)


def main(input_image_path, output_dir="output", filter_type="ideal", show_output=False):
    """
    主处理流程：比较频域高通滤波与 Sobel 边缘检测
    参数：
        input_image_path: 输入图像路径
        output_dir: 输出目录
        filter_type: 高通滤波器类型 ("ideal" 或 "gaussian")
        show_output: 是否显示输出图片
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- 1. 加载图像 -----
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)
    img_uint8 = img.astype(np.uint8)

    # ----- 2. 计算自适应截止频率（基于频谱能量95%）-----
    fft_orig = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_orig)
    D0 = compute_cutoff_frequency(fft_shifted, energy_percent=0.95)
    print(f"自适应截止频率 D0 = {D0:.1f} px (用于高通滤波)")

    # ----- 3. 构造并应用高通滤波器 -----
    if filter_type.lower() == "ideal":
        H_hp = ideal_highpass_filter(img.shape, D0)
        filter_name = "Ideal Highpass"
    else:
        H_hp = gaussian_highpass_filter(img.shape, D0)
        filter_name = "Gaussian Highpass"

    img_hp = apply_filter_and_reconstruct(fft_shifted, H_hp)

    # ----- 4. Sobel 边缘检测 -----
    img_sobel = sobel_edge_detection(img_uint8)

    # ----- 5. 生成对比结果图（1行3列布局：原图、高通、Sobel，外加底部说明）-----
    fig = plt.figure(figsize=(15, 7), facecolor='white')
    # 上移整行图片：增大 top 并适当增大 bottom 以整体上移但保持高度
    gs = fig.add_gridspec(1, 3, hspace=0.2, wspace=0.2,
                          left=0.05, right=0.95, top=0.94, bottom=0.20)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_hp   = fig.add_subplot(gs[0, 1])
    ax_sobel= fig.add_subplot(gs[0, 2])

    # 显示原图
    img_disp = normalize_display(img)
    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=12, fontweight='medium')
    ax_orig.axis('off')

    # 频域高通滤波结果
    ax_hp.imshow(img_hp, cmap='gray')
    ax_hp.set_title(f"Frequency Domain {filter_name}\nD₀={D0:.1f}px", fontsize=11, fontweight='medium')
    ax_hp.axis('off')

    # Sobel 结果
    ax_sobel.imshow(img_sobel, cmap='gray')
    ax_sobel.set_title("Sobel (Gradient Magnitude)", fontsize=11, fontweight='medium')
    ax_sobel.axis('off')

    # 添加底部说明文字（方法对比）
    text_str = (
        "Comparison Summary:\n\n"
        "• Frequency HPF: global edge enhancement.\n"
        "  - Ideal HPF: sharp cut-off, causes ringing.\n"
        "  - Gaussian HPF: smooth transition, no ringing.\n\n"
        "• Sobel: local gradient approximation,\n"
        "  simple and fast, thicker edges."
    )
    fig.text(0.5, 0.05, text_str, ha='center', fontsize=10,
             fontfamily='monospace', color='#1c1c1e',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='#f2f2f6', edgecolor='none'))

    fig.suptitle("Comparison: Frequency Domain Highpass vs. Sobel Edge Detection",
                 fontsize=14, fontweight='semibold', y=0.96)

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
    parser = argparse.ArgumentParser(description="频域高通滤波 vs Sobel 边缘检测对比")
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
            else:
                raise FileNotFoundError(f"未找到图像: {args.input}")
    main(args.input, args.output_dir, args.filter_type, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/hpf_vs_sobel.png")