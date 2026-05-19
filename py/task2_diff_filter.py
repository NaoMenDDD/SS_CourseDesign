'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-13 22:00:41
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-19 17:08:52
Description: 任务二：频域差分滤波器 - 二维一阶差分（梯度）

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
    """加载图像，转换为灰度图（0-255范围，float32）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像，请检查格式（支持 .bmp .jpg .png）: {image_path}")
    return img.astype(np.float32)


def compute_fft_spectrum(img):
    """计算二维FFT，返回频移后的复数频谱"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def freq_differential_filters(shape):
    """
    构造频域差分滤波器（基于傅里叶变换微分性质）
    返回：
        H_x: 水平差分滤波器（对x求偏导）即 ∂/∂x
        H_y: 垂直差分滤波器（对y求偏导）即 ∂/∂y
    公式：
        H_x(u,v) = j * 2π * u / M
        H_y(u,v) = j * 2π * v / N
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols) - ccol
    v = np.arange(rows) - crow
    U, V = np.meshgrid(u, v)

    H_x = 1j * 2 * np.pi * U / cols
    H_y = 1j * 2 * np.pi * V / rows
    return H_x, H_y


def apply_freq_filter(fft_shifted, filter_h):
    """应用频域滤波器，返回逆变换后的空域结果（实数）"""
    filtered_fft = fft_shifted * filter_h
    f_ishift = np.fft.ifftshift(filtered_fft)
    result = np.fft.ifft2(f_ishift)
    return np.real(result)


def magnitude_normalize(img):
    """将图像归一化到0-255范围（uint8）"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 1e-8:
        norm = (img - img_min) / (img_max - img_min) * 255.0
    else:
        norm = np.zeros_like(img)
    return norm.astype(np.uint8)


def visualize_filter_response(H_x):
    """
    可视化差分滤波器的幅度响应 |H|（取水平分量）
    返回对数幅度图（归一化到0-255）以及原始对数幅度数组（用于colorbar）
    """
    magnitude = np.abs(H_x)
    log_mag = np.log1p(magnitude)
    mag_norm = (log_mag - np.min(log_mag)) / (np.max(log_mag) - np.min(log_mag) + 1e-8)
    return (mag_norm * 255).astype(np.uint8), log_mag


def main(input_image_path, output_dir="output", show_output=False):
    """
    主处理流程：
    1. 加载灰度图像
    2. 计算FFT
    3. 构造频域差分滤波器 H_x, H_y
    4. 应用滤波得到水平梯度、垂直梯度
    5. 合成梯度幅值图
    6. 可视化滤波器频率响应
    7. 生成组合结果图（原图、水平梯度、垂直梯度、梯度幅值、滤波器响应）
    8. 保存组合图
    9. 按需显示输出结果
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- 1. 加载图像 -----
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)

    # ----- 2. FFT -----
    fft_shifted = compute_fft_spectrum(img)

    # ----- 3. 差分滤波器构造 -----
    H_x, H_y = freq_differential_filters(img.shape)

    # ----- 4. 应用滤波 -----
    grad_x_freq = apply_freq_filter(fft_shifted, H_x)   # ∂f/∂x
    grad_y_freq = apply_freq_filter(fft_shifted, H_y)   # ∂f/∂y

    # ----- 5. 梯度幅值合成 -----
    grad_mag_freq = np.sqrt(grad_x_freq**2 + grad_y_freq**2)

    # 归一化到0-255显示
    img_disp = magnitude_normalize(img)
    grad_x_norm = magnitude_normalize(grad_x_freq)
    grad_y_norm = magnitude_normalize(grad_y_freq)
    grad_mag_norm = magnitude_normalize(grad_mag_freq)

    # ----- 6. 滤波器响应可视化 -----
    filter_response, filter_log = visualize_filter_response(H_x)

    # ----- 7. 生成组合结果图（奥运五环式错位布局）-----
    was_interactive = plt.isinteractive()
    plt.ioff()
    fig = plt.figure(figsize=(18, 10), facecolor='white')

    ax_original = fig.add_axes([0.03, 0.510, 0.28, 0.34])
    ax_grad_x   = fig.add_axes([0.335, 0.510, 0.28, 0.34])
    ax_grad_y   = fig.add_axes([0.64, 0.510, 0.28, 0.34])
    ax_mag      = fig.add_axes([0.145, 0.080, 0.28, 0.34])
    ax_filter   = fig.add_axes([0.485, 0.080, 0.28, 0.34])

    # 显示图像
    ax_original.imshow(img_disp, cmap='gray')
    ax_original.set_title("Original Image", fontsize=13, fontweight='medium', pad=6)
    ax_original.axis('off')

    ax_grad_x.imshow(grad_x_norm, cmap='gray')
    ax_grad_x.set_title("Horizontal Gradient\n(∂f/∂x)", fontsize=13, fontweight='medium', pad=6)
    ax_grad_x.text(0.5, -0.06, "First-order difference along x highlights vertical edges.",
                   transform=ax_grad_x.transAxes, ha='center', fontsize=10.5, color='#8e8e93', style='italic')
    ax_grad_x.axis('off')

    ax_grad_y.imshow(grad_y_norm, cmap='gray')
    ax_grad_y.set_title("Vertical Gradient\n(∂f/∂y)", fontsize=13, fontweight='medium', pad=6)
    ax_grad_y.text(0.5, -0.06, "First-order difference along y highlights horizontal edges.",
                   transform=ax_grad_y.transAxes, ha='center', fontsize=10.5, color='#8e8e93', style='italic')
    ax_grad_y.axis('off')

    ax_mag.imshow(grad_mag_norm, cmap='gray')
    ax_mag.set_title("Gradient Magnitude\n|∇f|", fontsize=13, fontweight='medium', pad=6)
    ax_mag.text(0.5, -0.06, "Combines both directions of change; stronger edges produce larger values.",
                transform=ax_mag.transAxes, ha='center', fontsize=10.5, color='#8e8e93', style='italic')
    ax_mag.axis('off')

    # 频率响应带颜色条
    im = ax_filter.imshow(filter_response, cmap='hot')
    ax_filter.set_title("Frequency Response\n|H_diff(u,v)|", fontsize=13, fontweight='medium', pad=6)
    ax_filter.axis('off')
    cbar = fig.colorbar(im, ax=ax_filter, orientation='horizontal', fraction=0.05, pad=0.03)
    cbar.set_label('Gain (normalized log scale)', fontsize=10.5)

    # 总标题
    # 参考 ext3 的标题处理：减少顶部留白并让主标题位置更自然
    fig.suptitle("Frequency Domain Differential Filter (First-Order Difference)",
                 fontsize=20, fontweight='semibold', x=0.5, y=0.94, color='#1c1c1e')

    # 保存组合图
    output_combined = os.path.join(output_dir, "differential_filter_result.png")
    plt.savefig(output_combined, bbox_inches='tight', pad_inches=0.28, facecolor='white', dpi=200)
    plt.close(fig)
    if was_interactive:
        plt.ion()
    if show_output:
        # 直接显示保存后的图片
        saved_img = cv2.imread(output_combined, cv2.IMREAD_COLOR)
        if saved_img is not None:
            saved_img_rgb = cv2.cvtColor(saved_img, cv2.COLOR_BGR2RGB)
            plt.figure(facecolor='white')
            plt.imshow(saved_img_rgb, interpolation='nearest')
            plt.axis('off')
            plt.show()
            plt.close()
        else:
            print(f"⚠️ 无法读取输出图像进行显示: {output_combined}")
    print(f"✅ 结果图已保存至: {output_combined}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务二：频域差分滤波器 - 二维一阶差分（梯度）")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径（支持 .bmp .jpg .png），默认 img/house.bmp")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录，默认为 output")
    parser.add_argument("--show", action="store_true",
                        help="显示生成的输出图片")
    args = parser.parse_args()

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

    main(args.input, args.output_dir, show_output=args.show)
    print("\n处理完成！输出文件：")
    print(f" - {args.output_dir}/differential_filter_result.png")