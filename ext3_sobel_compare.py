'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-14 16:36:28
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-15 09:39:09
Description: 扩展任务三：频域高通滤波 vs Sobel vs Canny 边缘检测对比

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
    根据径向能量累计比例计算自适应截止频率
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
    # 理想高通：D <= D0 处为0，D > D0 处为1
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
    # 高斯高通 = 1 - 高斯低通
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
    # 频域滤波
    filtered = fft_shifted * filter_h
    # 逆FFT重建
    img_recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_recon = np.real(img_recon)
    img_recon = np.abs(img_recon)
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
    # 分别计算 x 和 y 方向梯度
    grad_x = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    # 计算梯度幅度
    mag = np.sqrt(grad_x**2 + grad_y**2)
    # 归一化到 0-255
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0
    return mag_norm.astype(np.uint8)


def canny_edge_detection(img_uint8, low_threshold=None, high_threshold=None):
    """
    使用 Canny 算子进行边缘检测
    参数：
        img_uint8: 输入灰度图（uint8）
        low_threshold: 低阈值（若为None则自动计算）
        high_threshold: 高阈值（若为None则自动计算）
    返回：
        二值边缘图（uint8）
    """
    if low_threshold is None or high_threshold is None:
        # 自动阈值计算：基于中位数
        sigma = 0.33
        v = np.median(img_uint8)
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(img_uint8, low, high)
    else:
        edges = cv2.Canny(img_uint8, low_threshold, high_threshold)
    return edges


def normalize_display(img):
    """
    将图像归一化到 0-255 范围用于显示
    参数：
        img: 输入图像数组
    返回：
        归一化后的图像（uint8）
    """
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(img).astype(np.uint8)


def main(input_image_path, output_dir="output", filter_type="ideal", canny_low=None, canny_high=None, show_output=False):
    """
    主处理流程：比较频域高通滤波 vs Sobel vs Canny 三种边缘检测方法
    参数：
        input_image_path: 输入图像路径
        output_dir: 输出目录
        filter_type: 高通滤波器类型 ("ideal" 或 "gaussian")
        canny_low: Canny 低阈值（若为None则自动计算）
        canny_high: Canny 高阈值（若为None则自动计算）
        show_output: 是否显示输出图片
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- 1. 加载图像 -----
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)
    img_uint8 = img.astype(np.uint8)

    # ----- 2. 计算自适应截止频率 -----
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

    # ----- 4. 应用三种边缘检测方法 -----
    img_hp = apply_filter_and_reconstruct(fft_shifted, H_hp)
    img_sobel = sobel_edge_detection(img_uint8)
    img_canny = canny_edge_detection(img_uint8, canny_low, canny_high)

    # ----- 5. 生成对比结果图 -----
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.15,
                          left=0.08, right=0.92, top=0.86, bottom=0.10)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_hp = fig.add_subplot(gs[0, 1])
    ax_sobel = fig.add_subplot(gs[0, 2])
    ax_canny = fig.add_subplot(gs[1, 1])
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis('off')

    # 原图
    img_disp = normalize_display(img)
    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=12, fontweight='medium')
    ax_orig.axis('off')

    # 频域高通滤波结果
    ax_hp.imshow(img_hp, cmap='gray')
    ax_hp.set_title(f"Frequency Domain {filter_name}\nD₀={D0:.1f}px", fontsize=11, fontweight='medium')
    ax_hp.axis('off')

    # Sobel 边缘检测结果
    ax_sobel.imshow(img_sobel, cmap='gray')
    ax_sobel.set_title("Sobel (Gradient Magnitude)", fontsize=11, fontweight='medium')
    ax_sobel.axis('off')

    # Canny 边缘检测结果
    ax_canny.imshow(img_canny, cmap='gray')
    canny_param = f"auto (median={np.median(img_uint8):.0f})" if (canny_low is None) else f"low={canny_low}, high={canny_high}"
    ax_canny.set_title(f"Canny Edge Detection\n{canny_param}", fontsize=11, fontweight='medium')
    ax_canny.axis('off')

    # 方法对比说明文本
    ax_legend.axis('off')
    text_str = (
        "Comparison Summary:\n\n"
        "• Frequency HPF: global edge enhancement,\n  may cause ringing (ideal)\n  or smooth (Gaussian).\n\n"
        "• Sobel: simple gradient magnitude,\n  response to all edges, thicker lines.\n\n"
        "• Canny: non-maximum suppression +\n  double threshold, thin and clean edges,\n  less noise."
    )
    ax_legend.text(0.05, 0.5, text_str, transform=ax_legend.transAxes, fontsize=10,
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='#f2f2f6', edgecolor='none'))

    fig.suptitle("Comparison: Frequency Domain Highpass vs. Sobel vs. Canny", fontsize=16, fontweight='semibold', y=0.94)
    fig.text(0.5, 0.03, "Highpass enhances edges globally; Sobel gives thick gradient; Canny produces thin, accurate edges.",
             fontsize=9, ha='center', color='#8e8e93', style='italic')

    # ----- 6. 保存结果 -----
    output_path = os.path.join(output_dir, "hpf_vs_sobel_vs_canny.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.28, facecolor='white', dpi=200)
    plt.close(fig)

    # ----- 7. 可选：显示结果 -----
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
    parser = argparse.ArgumentParser(description="扩展任务三增强版：频域高通滤波 vs Sobel vs Canny 边缘检测")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--filter_type", type=str, default="ideal", choices=["ideal", "gaussian"],
                        help="高通滤波器类型：ideal 或 gaussian，默认 ideal")
    parser.add_argument("--canny_low", type=int, default=None,
                        help="Canny低阈值（若不指定则自动计算）")
    parser.add_argument("--canny_high", type=int, default=None,
                        help="Canny高阈值（若不指定则自动计算）")
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
    main(args.input, args.output_dir, args.filter_type, args.canny_low, args.canny_high, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/hpf_vs_sobel_vs_canny.png")