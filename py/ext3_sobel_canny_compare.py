'''
Author: NaoMenDDD 2017954808@qq.com
Date: 2026-05-19 22:00:00
LastEditors: NaoMenDDD 2017954808@qq.com
Description: 扩展任务三子任务：Sobel vs Canny 边缘检测对比

对比两种边缘提取方法：
- 路径A (Sobel)：原图 → Sobel 算子 → 梯度幅值
- 路径B (Canny)：原图 → 高斯平滑 → 梯度计算 → 非极大值抑制 → 双阈值连接

输出组合对比图，包含原图、Sobel 结果、Canny 结果及方法说明。
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
        # 自动阈值计算：基于中位数的常用启发式（sigma=0.33）
        sigma = 0.33
        v = np.median(img_uint8)
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(img_uint8, low, high)
    else:
        edges = cv2.Canny(img_uint8, low_threshold, high_threshold)
    return edges


def normalize_display(img):
    """将任意图像归一化到 0-255 范围用于显示"""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(img).astype(np.uint8)


def main(input_image_path, output_dir="output", canny_low=None, canny_high=None, show_output=False):
    """
    主处理流程：比较 Sobel 与 Canny 边缘检测
    参数：
        input_image_path: 输入图像路径
        output_dir: 输出目录
        canny_low: Canny 低阈值（若为None则自动计算）
        canny_high: Canny 高阈值（若为None则自动计算）
        show_output: 是否显示输出图片
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- 1. 加载图像 -----
    print(f"加载图像: {input_image_path}")
    img = load_grayscale_image(input_image_path)
    img_uint8 = img.astype(np.uint8)

    # ----- 2. Sobel 边缘检测 -----
    img_sobel = sobel_edge_detection(img_uint8)

    # ----- 3. Canny 边缘检测 -----
    img_canny = canny_edge_detection(img_uint8, canny_low, canny_high)
    canny_param = f"auto (median={np.median(img_uint8):.0f})" if (canny_low is None) else f"low={canny_low}, high={canny_high}"

    # ----- 4. 生成对比结果图（1行3列布局：原图、Sobel、Canny，外加底部说明）-----
    fig = plt.figure(figsize=(15, 7), facecolor='white')
    gs = fig.add_gridspec(1, 3, hspace=0.2, wspace=0.2,
                          left=0.05, right=0.95, top=0.94, bottom=0.20)

    ax_orig   = fig.add_subplot(gs[0, 0])
    ax_sobel  = fig.add_subplot(gs[0, 1])
    ax_canny  = fig.add_subplot(gs[0, 2])

    # 显示原图
    img_disp = normalize_display(img)
    ax_orig.imshow(img_disp, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=12, fontweight='medium')
    ax_orig.axis('off')

    # Sobel 结果
    ax_sobel.imshow(img_sobel, cmap='gray')
    ax_sobel.set_title("Sobel (Gradient Magnitude)", fontsize=11, fontweight='medium')
    ax_sobel.axis('off')

    # Canny 结果
    ax_canny.imshow(img_canny, cmap='gray')
    ax_canny.set_title(f"Canny Edge Detection\n{canny_param}", fontsize=11, fontweight='medium')
    ax_canny.axis('off')

    # 添加底部说明文字
    text_str = (
        "Comparison Summary:\n\n"
        "• Sobel: simple gradient magnitude,\n"
        "  fast but edges are thicker and noisy.\n\n"
        "• Canny: multi-stage optimization:\n"
        "  Gaussian smoothing → gradient → non-maximum suppression\n"
        "  → double threshold → thin, clean, and accurate edges."
    )
    fig.text(0.5, 0.05, text_str, ha='center', fontsize=10,
             fontfamily='monospace', color='#1c1c1e',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='#f2f2f6', edgecolor='none'))

    fig.suptitle("Comparison: Sobel vs. Canny Edge Detection",
                 fontsize=14, fontweight='semibold', y=0.96)

    # ----- 5. 保存结果 -----
    output_path = os.path.join(output_dir, "sobel_vs_canny.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.28, facecolor='white', dpi=200)
    plt.close(fig)

    # ----- 6. 可选显示 -----
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
    parser = argparse.ArgumentParser(description="Sobel vs Canny 边缘检测对比")
    parser.add_argument("--input", "-i", type=str, default="img/house.bmp",
                        help="输入图像路径")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="输出目录")
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
    main(args.input, args.output_dir, args.canny_low, args.canny_high, args.show)
    print(f"\n完成！输出文件: {args.output_dir}/sobel_vs_canny.png")