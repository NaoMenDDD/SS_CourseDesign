'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-19 19:29:59
LastEditors: NaoMenDDD 2017954808@qq.com
LastEditTime: 2026-05-19 21:38:37
Description: 对比 Sobel 算子不同卷积核大小（3、7、11）的边缘检测效果，生成组合对比图

Copyright (c) 2026 by NaoMenDDD, All Rights Reserved. 
'''


import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# 设置 matplotlib 全局样式
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
    """加载灰度图像（返回 uint8）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img


def sobel_gradient_magnitude(img_uint8, ksize):
    """
    计算 Sobel 梯度幅度图
    参数：
        img_uint8: 输入灰度图像（uint8）
        ksize: 卷积核大小，必须为奇数（1,3,7,11）
    返回：
        归一化后的梯度幅度图（uint8，0-255）
    """
    # 分别计算 x 和 y 方向的梯度（使用浮点类型避免截断）
    grad_x = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=ksize)
    # 计算梯度幅度
    mag = np.sqrt(grad_x**2 + grad_y**2)
    # 归一化到 0-255
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def main(input_image_path, output_dir="output", show_output=False):
    """
    主流程：加载图像，分别用 ksize=3,7,11 计算 Sobel 边缘，生成组合对比图
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载图像
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)

    # 计算不同核大小的 Sobel 边缘
    sobel_3 = sobel_gradient_magnitude(img, ksize=3)
    sobel_7 = sobel_gradient_magnitude(img, ksize=7)
    sobel_11 = sobel_gradient_magnitude(img, ksize=11)

    # 生成组合图（2 行 2 列布局：原图，3x3，7x7，11x11）
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.2,
                          left=0.25, right=0.75, top=0.90, bottom=0.10)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_ksize3 = fig.add_subplot(gs[0, 1])
    ax_ksize7 = fig.add_subplot(gs[1, 0])
    ax_ksize11 = fig.add_subplot(gs[1, 1])

    # 显示原图
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=12, fontweight='medium')
    ax_orig.axis('off')

    # 显示 Sobel（核大小 3）
    ax_ksize3.imshow(sobel_3, cmap='gray')
    ax_ksize3.set_title("Sobel (ksize=3)\n(standard, fine edges)", fontsize=12, fontweight='medium')
    ax_ksize3.axis('off')

    # 显示 Sobel（核大小 7）
    ax_ksize7.imshow(sobel_7, cmap='gray')
    ax_ksize7.set_title("Sobel (ksize=7)\n(slightly smoother, thicker edges)", fontsize=12, fontweight='medium')
    ax_ksize7.axis('off')

    # 显示 Sobel（核大小 11）
    ax_ksize11.imshow(sobel_11, cmap='gray')
    ax_ksize11.set_title("Sobel (ksize=11)\n(stronger smoothing, thicker edges)", fontsize=12, fontweight='medium')
    ax_ksize11.axis('off')
    ax_ksize7.set_title("Sobel (ksize=7)\n(stronger smoothing, thicker edges)", fontsize=12, fontweight='medium')
    ax_ksize7.axis('off')

    # 总标题及说明
    fig.suptitle("Sobel Edge Detection: Effect of Kernel Size", fontsize=16, fontweight='semibold', y=0.96)
    fig.text(0.5, 0.03, "Larger kernel provides stronger smoothing but may lose fine details and thicken edges.",
             fontsize=10, ha='center', color='#8e8e93', style='italic')

    # 保存结果
    output_path = os.path.join(output_dir, "sobel_kernel_comparison.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.28, facecolor='white', dpi=200)
    plt.close(fig)

    # 可选显示
    if show_output:
        saved = cv2.imread(output_path, cv2.IMREAD_COLOR)
        if saved is not None:
            saved_rgb = cv2.cvtColor(saved, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(saved_rgb)
            plt.axis('off')
            plt.show()

    print(f"✅ Sobel 核大小对比图已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比 Sobel 算子不同卷积核大小（3、7、11）的边缘检测效果")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径（支持 .bmp .jpg .png），默认 img/house.bmp")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录，默认为 output")
    parser.add_argument("--show", action="store_true", help="显示生成的输出图片")
    args = parser.parse_args()

    # 检查默认图像是否存在，如果不存在则尝试 img 文件夹下的第一个图片
    if not os.path.exists(args.input):
        img_folder = Path("img")
        if img_folder.exists() and img_folder.is_dir():
            imgs = list(img_folder.glob("*.bmp")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
            if imgs:
                args.input = str(imgs[0])
                print(f"默认图像不存在，自动选择: {args.input}")
            else:
                raise FileNotFoundError(f"未找到图像: {args.input}")
    main(args.input, args.output_dir, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/sobel_kernel_comparison.png")