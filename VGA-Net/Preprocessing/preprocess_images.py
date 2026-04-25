# -*- coding: utf-8 -*-
"""
DRIVE 数据集预处理（修复版）
使用 FOV mask 精确处理视野内外区域，避免四个角出现橙色 artifacts。
"""
import cv2
import numpy as np
import os


def apply_clahe(image, mask):
    """仅对 mask 内部区域应用 CLAHE"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # CLAHE 默认会处理整图，但我们只替换 mask 内部
    clahe_l_full = clahe.apply(l_channel)
    l_channel = np.where(mask == 255, clahe_l_full, l_channel)
    
    clahe_lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2BGR)


def unsharp_mask(image, mask, sigma=1.0, strength=1.5):
    """仅对 mask 内部区域应用非锐化掩模"""
    b, g, r = cv2.split(image)
    
    for ch in [b, g, r]:
        blurred = cv2.GaussianBlur(ch, (0, 0), sigma)
        sharpened = cv2.addWeighted(ch, 1.0 + strength, blurred, -strength, 0)
        ch[:] = np.where(mask == 255, sharpened, ch)
    
    return cv2.merge((b, g, r))


def preprocess_image(image, mask):
    """
    预处理单张图像：
    1. 仅对 FOV mask 内部应用 CLAHE
    2. 仅对 FOV mask 内部应用 unsharp mask
    3. FOV 外部保持原始黑色
    """
    # CLAHE 增强（仅限 mask 内部）
    clahe_image = apply_clahe(image, mask)
    # 锐化（仅限 mask 内部）
    result = unsharp_mask(clahe_image, mask)
    return result


def process_split(split_name, images_dir, masks_dir, output_dir):
    """处理训练集或测试集"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    print(f"处理 {split_name}: 找到 {len(image_files)} 张图像")
    
    for file_name in image_files:
        base_name = file_name.replace('.tif', '')
        mask_name = f"{base_name}_mask.gif"
        
        img_path = os.path.join(images_dir, file_name)
        mask_path = os.path.join(masks_dir, mask_name)
        out_path = os.path.join(output_dir, file_name)
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"  跳过: 无法读取图像 {img_path}")
            continue
        if mask is None:
            print(f"  跳过: 无法读取 mask {mask_path}")
            continue
        
        processed = preprocess_image(image, mask)
        cv2.imwrite(out_path, processed)
    
    print(f"  完成: 结果保存到 {output_dir}")


if __name__ == '__main__':
    base_dir = '/root/autodl-tmp/VGA-Net/DRIVE'
    
    # 处理训练集
    process_split(
        'training',
        os.path.join(base_dir, 'training', 'images'),
        os.path.join(base_dir, 'training', 'mask'),
        os.path.join(base_dir, 'training', 'preprocessed')
    )
    
    # 处理测试集
    process_split(
        'test',
        os.path.join(base_dir, 'test', 'images'),
        os.path.join(base_dir, 'test', 'mask'),
        os.path.join(base_dir, 'test', 'preprocessed')
    )
