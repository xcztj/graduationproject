# -*- coding: utf-8 -*-
"""VGA-Net 推理模块"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from Model.VGA_Net import FinalNetwork


class VGAInferencer:
    """VGA-Net 推理器"""
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = FinalNetwork()
        self.model.to(self.device)
        
        # 加载权重
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型加载成功: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model.eval()
        self.input_size = (584, 565)
    
    def preprocess(self, image_path):
        """预处理单张图像"""
        # 读取图像 (BGR)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # resize 到模型输入尺寸
        img_resized = cv2.resize(img_rgb, self.input_size[::-1])  # (W, H)
        
        # 归一化到 [0, 1]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        return img_tensor, img_rgb
    
    def load_mask(self, mask_path, img_shape):
        """加载并预处理 mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        # resize 到原图尺寸
        mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
        # 二值化
        mask = (mask > 127).astype(np.float32)
        return mask
    
    @torch.no_grad()
    def predict(self, image_path):
        """对单张图像进行推理"""
        img_tensor, img_rgb = self.preprocess(image_path)
        
        # 前向传播
        output = self.model(img_tensor)
        
        # 取出预测结果
        pred = output.squeeze().cpu().numpy()  # (H, W)
        
        # 二值化
        pred_binary = (pred > 0.5).astype(np.float32)
        
        return img_rgb, pred, pred_binary
    
    def compute_metrics(self, pred_binary, mask):
        """计算评估指标"""
        if mask is None:
            return None
        
        pred_flat = pred_binary.flatten()
        mask_flat = mask.flatten()
        
        # Accuracy
        acc = np.mean(pred_flat == mask_flat)
        
        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(mask_flat, pred_flat).ravel()
        except ValueError:
            # 全为正或全为负的情况
            return {'Accuracy': acc, 'Dice': 0.0, 'SE': 0.0, 'SP': 0.0, 'MCC': 0.0}
        
        # Sensitivity (Recall)
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Dice
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        
        # MCC
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator if denominator > 0 else 0.0
        
        return {
            'Accuracy': acc,
            'Dice': dice,
            'SE': se,
            'SP': sp,
            'MCC': mcc
        }
    
    def predict_batch(self, image_dir, mask_dir=None, output_dir=None):
        """批量推理"""
        import glob
        
        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
        results = []
        
        for img_path in image_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            img_rgb, pred_prob, pred_binary = self.predict(img_path)
            
            # 加载 mask（如果存在）
            mask = None
            if mask_dir and os.path.exists(mask_dir):
                mask_path = os.path.join(mask_dir, f"{basename}_manual1.gif")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(mask_dir, f"{basename}_mask.gif")
                if os.path.exists(mask_path):
                    mask = self.load_mask(mask_path, img_rgb.shape)
            
            # 计算指标
            metrics = self.compute_metrics(pred_binary, mask) if mask is not None else None
            
            results.append({
                'name': basename,
                'image': img_rgb,
                'pred_prob': pred_prob,
                'pred_binary': pred_binary,
                'mask': mask,
                'metrics': metrics
            })
            
            # 保存结果
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                pred_save = (pred_binary * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"{basename}_pred.png"), pred_save)
        
        return results
