import torch
import torch_dct as dct
import numpy as np
from torchvision import transforms
from models.dct_layer import DCT2DLayer, MyDCT2DLayer, MyInverseDCT2DLayer

import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import seaborn as sns


def load_and_preprocess_image(image_path):
    """
    加载JPG图片并转换为PyTorch tensor
    """
    # 使用PIL加载图像
    img_rgb = Image.open(image_path)
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    print(f"原始图像尺寸: {img_rgb.size}")
    
    # 转换为numpy数组
    img_array = np.array(img, dtype=np.float32)
    img_array_rgb = np.array(img_rgb, dtype=np.float32)
    img_r, img_g, img_b = np.split(img_array_rgb, 3, axis=-1)

    # img_array = np.random.rand(*img.size)*255
    # img_array_rgb = np.random.rand(*img_rgb.size)*255
    # img_r, img_g, img_b = np.random.rand(*img_rgb.size)*255, np.random.rand(*img_rgb.size)*255, np.random.rand(*img_rgb.size)*255
    
    # 转换为PyTorch tensor并添加batch维度
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
    
    return img_tensor, img_array, img_r, img_g, img_b

def apply_dct_and_visualize(image_path, block_size=8):
    """
    对图像进行DCT变换并可视化结果
    """
    # 1. 加载和预处理图像
    img_tensor, original_img, img_r, img_g, img_b = load_and_preprocess_image(image_path)
    H, W = original_img.shape
    
    print(f"图像张量形状: {img_tensor.shape}")
    print(f"像素值范围: [{original_img.min():.1f}, {original_img.max():.1f}]")
    
    # 2. 如果图像太大，可以选择分割成块进行处理，或者调整大小
    # 这里我们直接对整个图像进行DCT，或者调整到合适的大小
    max_size = 512
    if H > max_size or W > max_size:
        # 调整图像大小以便更好地可视化
        resize = transforms.Resize((max_size, max_size))
        img_tensor = resize(img_tensor.unsqueeze(0)).squeeze(0)
        H, W = max_size, max_size
        print(f"调整后图像尺寸: {H}x{W}")
    
    # 3. 应用二维DCT变换
    dctnet = MyDCT2DLayer(H, W)
    # assert img_tensor.ndim == 4, f"Image must be 4 dims! Got {img_tensor.ndim} dims"
    dct_result = dctnet(img_tensor)
    # dct_result = dct.dct_2d(img_tensor)
    print(f"DCT结果张量形状: {dct_result.shape}")
    print(f"DCT系数范围: [{dct_result.min():.2f}, {dct_result.max():.2f}]")
    
    # 4. 对DCT系数取绝对值并转换为numpy数组用于可视化
    dct_magnitude = torch.abs(dct_result).squeeze().numpy()
    dct_magnitude = (dct_magnitude - dct_magnitude.min()) /  (dct_magnitude.max() - dct_magnitude.min() + 1e-8)
    
    # 5. 创建可视化图表
    fig, axes = plt.subplots(2, 4, figsize=(30, 12))
    
    # 5.1 原始图像
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title(f'original Gray {W}x{H}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_r, cmap='Reds')
    axes[0, 1].set_title(f'original Red {W}x{H}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_g, cmap='Greens')
    axes[0, 2].set_title(f'original Green {W}x{H}')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(img_b, cmap='Blues')
    axes[0, 3].set_title(f'original Blue {W}x{H}')
    axes[0, 3].axis('off')
    
    # 5.2 DCT系数的热力图（线性尺度）
    im1 = axes[1, 0].imshow(dct_magnitude[int(W/128):, int(H/128):], cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_title(f'DCT heatmap \n[{int(W/128)}:, {int(H/128)}:] \n(linear scale)')
    axes[1, 0].set_xlabel('horizontal freq')
    axes[1, 0].set_ylabel('vertical freq')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 5.3 DCT系数的热力图（对数尺度）- 更好地显示动态范围
    # 添加小值避免log(0)
    dct_log = np.log10(dct_magnitude + 1e-10)
    im2 = axes[1, 1].imshow(dct_log, cmap='viridis', aspect='auto')
    axes[1, 1].set_title(f'DCT heatmap \n[{int(W/128)}:, {int(H/128)}:] \n(log scale)')
    axes[1, 1].set_xlabel('horizontal freq')
    axes[1, 1].set_ylabel('vertical freq')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 5.4 使用seaborn绘制更精细的热力图（显示左上角低频区域）
    # 只显示低频部分（左上角64x64区域）
    crop_size = int(min(300, H, W))
    # dct_cropped = dct_magnitude[-crop_size:, -crop_size:]
    dct_cropped = dct_magnitude[int(W/2):crop_size, int(H/2):crop_size]

    ax4 = axes[1, 2]
    im4 = ax4.imshow(dct_cropped, cmap='cool', aspect='auto')
    ax4.set_title(f'low-freq area \n[{int(W/128)}:{crop_size}, {int(H/128)}:{crop_size}] \n(left-up{crop_size}x{crop_size})')
    ax4.set_xlabel('horizontal freq')
    ax4.set_ylabel('vertical freq')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = axes[1, 3]
    im5 = ax5.imshow(np.log10(dct_cropped + 1e-10), cmap='YlGnBu', aspect='auto')
    ax5.set_title(f'log-scale low-freq area \n[{int(W/128)}:{crop_size}, {int(H/128)}:{crop_size}] \n(left-up{crop_size}x{crop_size})')
    ax5.set_xlabel('horizontal freq')
    ax5.set_ylabel('vertical freq')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('./dct.png')
    plt.show()
    
    # 6. 打印DCT系数的统计信息
    print("\n=== DCT系数统计信息 ===")
    print(f"总能量 (系数平方和): {np.sum(dct_magnitude**2):.2f}")
    print(f"平均幅度: {np.mean(dct_magnitude):.4f}")
    print(f"最大幅度: {np.max(dct_magnitude):.4f}")
    print(f"最小幅度: {np.min(dct_magnitude):.4f}")
    
    # 计算能量分布
    total_energy = np.sum(dct_magnitude**2)
    
    # 左上角1/4区域的能量占比
    quarter_h, quarter_w = H//4, W//4
    low_freq_energy = np.sum(dct_magnitude[:quarter_h, :quarter_w]**2)
    low_freq_ratio = low_freq_energy / total_energy
    print(f"低频区域能量占比 (左上1/4): {low_freq_ratio*100:.2f}%")
    
    # 左上角1/8区域的能量占比
    eighth_h, eighth_w = H//8, W//8
    very_low_freq_energy = np.sum(dct_magnitude[:eighth_h, :eighth_w]**2)
    very_low_freq_ratio = very_low_freq_energy / total_energy
    print(f"极低频区域能量占比 (左上1/8): {very_low_freq_ratio*100:.2f}%")
    
    return dct_result, original_img

# 使用示例
if __name__ == "__main__":
    # 请将下面的路径替换为你的JPG图片路径
    dataset_path = "/home/ruihan/data/imagenet/train/"

    class_path = random.sample(os.listdir(dataset_path), 1)[0]
    class_path = os.path.join(dataset_path, class_path)

    image_path = random.sample(os.listdir(class_path), 1)[0]
    image_path = os.path.join(class_path, image_path)

    print("Image path: ", image_path)

    # image_path = "/home/ruihan/data/imagenet/train/n01530575/n01530575_489.JPEG"  # 替换为你的图片路径
    
    try:
        dct_coeff, original_image = apply_dct_and_visualize(image_path)
        
        # 可选：保存DCT系数图像
        # dct_magnitude = torch.abs(dct_coeff).squeeze().numpy()
        # plt.imsave('dct_heatmap.png', np.log10(dct_magnitude + 1e-10), cmap='viridis')
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
        print("请确保：")
        print("1. 图片路径正确")
        print("2. 图片文件存在")
        print("3. 你有读取该文件的权限")
    except Exception as e:
        print(f"处理图像时发生错误: {e}")