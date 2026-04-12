# -*- coding: utf-8 -*-

#-------------------------------
#预处理版本 v1（带 replace_black_area）
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 替换图像中黑色区域的函数
def replace_black_area(image):
    # 将图像转换为 RGB 颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 计算每个通道的平均颜色
    average_red = np.mean(image[:, :, 0])
    average_green = np.mean(image[:, :, 1])
    average_blue = np.mean(image[:, :, 2])

    # 创建黑色区域的掩码
    black_mask = np.all(image < [average_red, average_green, average_blue], axis=2)

    # 用平均颜色替换黑色区域
    image[black_mask] = [average_red, average_green, average_blue]

    # 对图像进行模糊处理以平滑边缘
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 将图像转换回 BGR 颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 返回修改后的图像
    return image

# 应用 CLAHE 增强图像对比度的函数
def apply_clahe(image):
    # 将图像转换为 LAB 颜色空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 将 LAB 图像拆分为 L、A 和 B 通道
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 对 L 通道应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)

    # 将 CLAHE 增强后的 L 通道与原始 A 和 B 通道合并
    clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

    # 将 CLAHE 增强后的 LAB 图像转换回 BGR 颜色空间
    clahe_bgr_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

    return clahe_bgr_image

# 应用非锐化掩模来锐化图像的函数
def unsharp_mask(image, sigma=1.0, strength=1.5):
    # 将图像拆分为 B、G 和 R 通道
    b_channel, g_channel, r_channel = cv2.split(image)

    # 对每个通道应用高斯模糊
    blurred_b = cv2.GaussianBlur(b_channel, (0, 0), sigma)
    blurred_g = cv2.GaussianBlur(g_channel, (0, 0), sigma)
    blurred_r = cv2.GaussianBlur(r_channel, (0, 0), sigma)

    # 计算每个通道的锐化图像
    sharp_b = cv2.addWeighted(b_channel, 1.0 + strength, blurred_b, -strength, 0)
    sharp_g = cv2.addWeighted(g_channel, 1.0 + strength, blurred_g, -strength, 0)
    sharp_r = cv2.addWeighted(r_channel, 1.0 + strength, blurred_r, -strength, 0)

    # 将锐化后的通道合并为单个图像
    sharp_image = cv2.merge((sharp_b, sharp_g, sharp_r))

    return sharp_image

# 将黑色区域替换回原始图像的函数
def replace_black_area_back(original_image, modified_image):
    # 将图像转换为 RGB 颜色空间
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

    # 计算原始图像每个通道的平均颜色
    average_red_original = np.mean(original_image[:, :, 0])
    average_green_original = np.mean(original_image[:, :, 1])
    average_blue_original = np.mean(original_image[:, :, 2])

    # 为修改后的图像创建黑色区域的掩码
    black_mask = np.all(modified_image < [average_red_original, average_green_original, average_blue_original], axis=2)

    # 用原始值替换修改后图像中的黑色区域
    modified_image[black_mask] = original_image[black_mask]

    # 将图像转换回 BGR 颜色空间
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)

    # 返回黑色区域已恢复的修改后图像
    return modified_image

# 加载原始图像
original_image = cv2.imread('23_training.jpg')

# 按顺序应用每个步骤
modified_image = replace_black_area(original_image)
clahe_image = apply_clahe(modified_image)
unsharp_masked_image = unsharp_mask(clahe_image)
restored_image = replace_black_area_back(original_image, unsharp_masked_image)

# 显示所有图像
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('原始图像')

axs[0, 1].imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('黑色区域已替换的图像')

axs[0, 2].imshow(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('CLAHE 增强后的图像')

axs[1, 0].imshow(cv2.cvtColor(unsharp_masked_image, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('非锐化掩模后的图像')

axs[1, 1].imshow(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('恢复后的图像')

# 隐藏坐标轴
for ax in axs.flat:
    ax.axis('off')

# 显示图像
plt.tight_layout()
plt.show()


#----------------------------
#预处理版本 v2（不带 replace_black_area）
# 加载原始图像
original_image = cv2.imread('23_training.jpg')

# 将图像转换为 LAB 颜色空间
lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)

# 将 LAB 图像拆分为 L、A 和 B 通道
l_channel, a_channel, b_channel = cv2.split(lab_image)

# 对 L 通道应用 CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_l_channel = clahe.apply(l_channel)

# 将 CLAHE 增强后的 L 通道与原始 A 和 B 通道合并
clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

# 将 CLAHE 增强后的 LAB 图像转换回 BGR 颜色空间
clahe_bgr_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

# 显示原始图像和 CLAHE 增强后的图像
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(clahe_bgr_image, cv2.COLOR_BGR2RGB))
plt.title('CLAHE 增强后的图像')
plt.axis('off')

# 应用非锐化掩模来锐化图像的函数
def unsharp_mask(image, sigma=1.0, strength=1.5):
    # 将图像拆分为 B、G 和 R 通道
    b_channel, g_channel, r_channel = cv2.split(image)

    # 对每个通道应用高斯模糊
    blurred_b = cv2.GaussianBlur(b_channel, (0, 0), sigma)
    blurred_g = cv2.GaussianBlur(g_channel, (0, 0), sigma)
    blurred_r = cv2.GaussianBlur(r_channel, (0, 0), sigma)

    # 计算每个通道的锐化图像
    sharp_b = cv2.addWeighted(b_channel, 1.0 + strength, blurred_b, -strength, 0)
    sharp_g = cv2.addWeighted(g_channel, 1.0 + strength, blurred_g, -strength, 0)
    sharp_r = cv2.addWeighted(r_channel, 1.0 + strength, blurred_r, -strength, 0)

    # 将锐化后的通道合并为单个图像
    sharp_image = cv2.merge((sharp_b, sharp_g, sharp_r))

    return sharp_image

# 对 CLAHE 增强后的图像应用非锐化掩模
unsharp_masked_image = unsharp_mask(clahe_bgr_image)

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(unsharp_masked_image, cv2.COLOR_BGR2RGB))
plt.title('非锐化掩模后的图像')
plt.axis('off')
plt.tight_layout()
plt.show()
#----------------------------
#保存结果
import os

# 包含数据集的目录
dataset_dir = 'yeganeh/dataset/DRIVE'

# 保存预处理图像的目录
output_dir = 'yeganeh/extracted/patches'

# 如果输出目录不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 列出数据集目录中的所有文件
file_list = os.listdir(dataset_dir)

# 遍历数据集目录中的每个文件
for file_name in file_list:
    # 读取原始图像
    original_image = cv2.imread(os.path.join(dataset_dir, file_name))
    
    # 按顺序应用预处理的每个步骤
    modified_image = replace_black_area(original_image)
    clahe_image = apply_clahe(modified_image)
    unsharp_masked_image = unsharp_mask(clahe_image)
    restored_image = replace_black_area_back(original_image, unsharp_masked_image)
    
    # 保存预处理后的图像
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, restored_image)
