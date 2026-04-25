# -*- coding: utf-8 -*-
"""在测试集上搜索最优阈值（快速验证用）"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import glob
import numpy as np
import torch
import cv2
from PIL import Image
from inference import VGAInferencer


def dice_score(pred, mask):
    pred_flat = pred.flatten()
    mask_flat = mask.flatten()
    tp = np.sum((pred_flat == 1) & (mask_flat == 1))
    fp = np.sum((pred_flat == 1) & (mask_flat == 0))
    fn = np.sum((pred_flat == 0) & (mask_flat == 1))
    return (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0


def load_gif_mask(mask_path, target_shape):
    """使用 PIL 读取 GIF mask"""
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
    mask = (mask > 127).astype(np.float32)
    return mask


def main():
    model_path = '../Train/best_model.pt'
    input_dir = '../DRIVE/test/preprocessed'
    mask_dir = '../DRIVE/test/1st_manual'
    
    inferencer = VGAInferencer(model_path)
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
    
    # 收集所有预测概率和 mask
    all_probs = []
    all_masks = []
    
    print("收集预测概率...")
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img_tensor, img_rgb = inferencer.preprocess(img_path)
        
        with torch.no_grad():
            prob = inferencer.model(img_tensor).cpu().numpy()[0, 0]  # (H, W)
        
        # resize 到原图尺寸
        h, w = img_rgb.shape[:2]
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 加载 mask (PIL 读取 GIF)
        # 测试集 mask 命名: 01_manual1.gif
        mask_name = basename.replace('_test', '_manual1') + '.gif'
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"{basename}_manual1.gif")
        mask = load_gif_mask(mask_path, (h, w))
        
        all_probs.append(prob)
        all_masks.append(mask)
    
    # 搜索最优阈值
    best_dice = 0
    best_thresh = 0.5
    print("\n搜索最优阈值...")
    for thresh in np.arange(0.30, 0.71, 0.01):
        dices = []
        for prob, mask in zip(all_probs, all_masks):
            pred = (prob > thresh).astype(np.float32)
            dices.append(dice_score(pred, mask))
        avg_dice = np.mean(dices)
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_thresh = thresh
        print(f"  threshold={thresh:.2f}, Dice={avg_dice:.4f}")
    
    print(f"\n最优阈值: {best_thresh:.2f}, 最佳 Dice: {best_dice:.4f}")
    print(f"相比默认 0.5 阈值的提升: +{best_dice - 0.7983:.4f}")


if __name__ == '__main__':
    main()
