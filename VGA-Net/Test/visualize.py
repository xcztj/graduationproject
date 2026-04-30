# -*- coding: utf-8 -*-
"""
血管分割可视化脚本
生成：原图 | Ground Truth | 预测结果 | 差异图
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from Train.dataset import ToTensor, DRIVEDataset
from Model.VGA_Net import FinalNetwork

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = FinalNetwork()
model.load_state_dict(torch.load('/root/autodl-tmp/VGA-Net/Train/best_model.pt', map_location=device))
model.to(device)
model.eval()

# 加载测试集
test_dataset = DRIVEDataset(
    root_dir='/root/autodl-tmp/VGA-Net/DRIVE/test',
    transform=ToTensor(),
    use_preprocessed=True,
    augment=False
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建输出目录
output_dir = '/root/autodl-tmp/VGA-Net/result'
os.makedirs(output_dir, exist_ok=True)

print(f"开始可视化，共 {len(test_dataset)} 张测试图像...")
print(f"结果保存到: {output_dir}")

def create_comparison(image, gt, pred, img_name, threshold=0.5):
    """
    创建四宫格可视化：
    [原图] [GT] [预测] [差异]
    """
    # 转为 numpy
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    gt_np = gt.squeeze().cpu().numpy()  # (H, W)
    pred_prob = pred.squeeze().cpu().numpy()  # (H, W)
    pred_bin = (pred_prob > threshold).astype(np.float32)
    
    # 归一化原图到 [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    h, w = img_np.shape[:2]
    
    # 创建 mask 彩色图（绿色=GT/预测，红色=差异）
    gt_color = np.zeros((h, w, 3), dtype=np.uint8)
    pred_color = np.zeros((h, w, 3), dtype=np.uint8)
    diff_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # GT: 白色血管
    gt_color[gt_np > 0.5] = [255, 255, 255]
    
    # 预测: 白色血管
    pred_color[pred_bin > 0.5] = [255, 255, 255]
    
    # 差异图:
    # TP (正确检出): 绿色
    # FP (误检): 红色  
    # FN (漏检): 蓝色
    tp = (gt_np > 0.5) & (pred_bin > 0.5)
    fp = (gt_np <= 0.5) & (pred_bin > 0.5)
    fn = (gt_np > 0.5) & (pred_bin <= 0.5)
    
    diff_color[tp] = [0, 255, 0]      # 绿色 = TP
    diff_color[fp] = [0, 0, 255]      # 红色 = FP
    diff_color[fn] = [255, 0, 0]      # 蓝色 = FN
    
    # 叠加到原图（半透明）
    overlay_tp = img_np.copy()
    overlay_tp[tp] = overlay_tp[tp] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay_fp = img_np.copy()
    overlay_fp[fp] = overlay_fp[fp] * 0.5 + np.array([0, 0, 255]) * 0.5
    overlay_fn = img_np.copy()
    overlay_fn[fn] = overlay_fn[fn] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # 合成差异图
    diff_overlay = img_np.copy()
    diff_overlay[tp] = diff_overlay[tp] * 0.6 + np.array([0, 255, 0]) * 0.4
    diff_overlay[fp] = diff_overlay[fp] * 0.6 + np.array([0, 0, 255]) * 0.4
    diff_overlay[fn] = diff_overlay[fn] * 0.6 + np.array([255, 0, 0]) * 0.4
    
    # 水平拼接四张图
    # 第一行: 原图 | GT | 预测 | 差异叠加
    row1 = np.hstack([img_np, gt_color, pred_color, diff_overlay])
    
    # 添加标题
    title_h = 40
    result = np.zeros((h + title_h, w * 4, 3), dtype=np.uint8)
    result[title_h:, :] = row1
    
    # 添加文字标题
    titles = ['Original', 'Ground Truth', 'Prediction', 'Diff (G=TP, R=FP, B=FN)']
    for i, title in enumerate(titles):
        x = i * w + w // 2 - len(title) * 6
        cv2.putText(result, title, (max(10, x), 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 保存
    save_path = os.path.join(output_dir, f'{img_name}_result.png')
    cv2.imwrite(save_path, result)
    
    return save_path

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        inputs = batch['image'].to(device)
        labels = batch['mask'].to(device)
        
        # 预测
        outputs = model(inputs)
        
        # 获取图像名称
        img_name = test_dataset.image_files[idx].replace('.tif', '')
        
        # 生成可视化
        save_path = create_comparison(inputs, labels, outputs, img_name, threshold=0.5)
        
        print(f"  [{idx+1}/{len(test_dataset)}] {img_name} -> saved")

print(f"\n全部完成！结果保存在: {output_dir}")
print(f"共生成 {len(test_dataset)} 张对比图")
