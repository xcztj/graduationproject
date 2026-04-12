# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

# 从图像中提取随机图像块的函数
def extract_random_patches(image, patch_size, num_patches):
    patches = []
    height, width, _ = image.shape
    for _ in range(num_patches):
        # 生成图像块左上角的随机坐标
        top_left_x = np.random.randint(0, width - patch_size)
        top_left_y = np.random.randint(0, height - patch_size)
        
        # 从图像中提取图像块
        patch = image[top_left_y:top_left_y+patch_size, top_left_x:top_left_x+patch_size]
        patches.append(patch)
    return patches

# 包含预处理图像的目录
preprocessed_dir = 'yeganeh/preprocessed/images'

# 保存提取图像块的目录
patches_dir = 'yeganeh/extracted/patches'

# 如果图像块目录不存在，则创建它
if not os.path.exists(patches_dir):
    os.makedirs(patches_dir)

# 图像块大小
patch_size = 48

# 从每张图像中提取的图像块数量
num_patches_per_image = 143

# 列出预处理目录中的所有预处理图像
file_list = os.listdir(preprocessed_dir)

# 遍历每个预处理图像
for file_name in file_list:
    # 读取预处理图像
    preprocessed_image = cv2.imread(os.path.join(preprocessed_dir, file_name))
    
    # 从预处理图像中提取随机图像块
    patches = extract_random_patches(preprocessed_image, patch_size, num_patches_per_image)
    
    # 保存提取的图像块
    for i, patch in enumerate(patches):
        patch_name = os.path.splitext(file_name)[0] + f'_patch_{i}.jpg'
        cv2.imwrite(os.path.join(patches_dir, patch_name), patch)
