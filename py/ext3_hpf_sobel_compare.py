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
    # 将频率索引中心化：fftshift 后直流分量在频谱中心，
    # 因此这里生成以中心为原点的坐标系（负频率/正频率对称）。
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)

    # 计算每个频率点到频谱中心的径向距离 D(u,v)。
    # 在理想高通中，低频（小于等于 D0）的分量被完全抑制（置为0），
    # 高频（大于 D0）完整保留（置为1）；因此这里先创建全 1 矩阵，
    # 再将中心半径内的点设为 0。
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
    # 同样使用中心化坐标系计算径向平方距离 D^2
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)
    D2 = U**2 + V**2

    # 高斯低通响应 H_lp = exp(-D^2 / (2 * D0^2))，在中心接近 1，
    # 随距离增大平滑衰减；高斯高通则为 1 - H_lp，得到平滑的高通响应，
    # 相比理想高通能显著减少振铃（ringing）伪影。
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
    # 频域相乘等价于空域的线性卷积或微分操作。
    # 假定传入的 fft_shifted 已做过 np.fft.fftshift，且 filter_h 的中心与之对齐。
    filtered = fft_shifted * filter_h

    # 在进行逆 FFT 前需要将频谱移回原始布局（ifftshift），
    # 否则 ifft2 会错误地解释频率排列。
    img_recon = np.fft.ifft2(np.fft.ifftshift(filtered))

    # 逆变换结果通常为复数（主要是数值误差或相位分量），取其实部作为重建图像。
    img_recon = np.real(img_recon)

    # 对于高通操作，重建后可能出现负值（因为滤掉了直流分量），
    # 此处取绝对值以便于可视化边缘强度；这一步是视觉化上的处理，
    # 若需保持符号信息可去掉 abs 操作并以合适方式显示。
    img_recon = np.abs(img_recon)

    # 将结果线性归一化到 0-255 便于显示与保存。
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
    # Sobel 算子是一个离散的微分算子，用于近似图像的空间梯度。
    # 这里使用 OpenCV 的 Sobel 实现，返回浮点型梯度值：
    # - grad_x: 对 x 方向（列）的一阶导近似，强调垂直边缘；
    # - grad_y: 对 y 方向（行）的一阶导近似，强调水平边缘。
    # 使用 cv2.CV_32F 以保留正负梯度信息并避免溢出。
    grad_x = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)

    # 梯度幅值是两个方向分量的欧氏范数，表示边缘强度。
    # 使用平方和开方可以合并两个方向的信息，得到单通道的边缘响应图。
    mag = np.sqrt(grad_x**2 + grad_y**2)

    # 为了可视化，将幅值线性归一化到 0-255 区间：
    # - 减去最小值并除以动态范围将其映射到 [0,1]
    # - 乘以 255 得到 8 位显示范围
    # + 通过 +1e-8 防止除以零的数值不稳定情况。
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0
    return mag_norm.astype(np.uint8)


def normalize_display(img):
    """将任意图像归一化到 0-255 范围用于显示"""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(img).astype(np.uint8)


def main(input_image_path, output_dir="output", show_output=False):
    """
    主处理流程：比较频域高通滤波与 Sobel 边缘检测
    参数：
        input_image_path: 输入图像路径
        output_dir: 输出目录
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
    parser = argparse.ArgumentParser(description="高通滤波 vs Sobel 边缘检测对比")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录")
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
    main(args.input, args.output_dir, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/hpf_vs_sobel.png")