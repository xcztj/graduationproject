# -*- coding: utf-8 -*-
"""
批量推理脚本（带 TTA - Test Time Augmentation）
支持水平翻转和垂直翻转的预测平均
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from inference import VGAInferencer


def compute_metrics(pred_binary, mask):
    pred_flat = pred_binary.flatten()
    mask_flat = mask.flatten()
    acc = np.mean(pred_flat == mask_flat)
    try:
        tn, fp, fn, tp = confusion_matrix(mask_flat, pred_flat).ravel()
        tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
    except ValueError:
        return {'Accuracy': acc, 'Dice': 0.0, 'SE': 0.0, 'SP': 0.0, 'MCC': 0.0}
    
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0.0
    
    return {'Accuracy': acc, 'Dice': dice, 'SE': se, 'SP': sp, 'MCC': mcc}


class TTAInferencer(VGAInferencer):
    """扩展 VGAInferencer 支持 TTA"""
    
    def predict_tta(self, image_path, flips=['none', 'hflip']):
        """
        TTA 预测：对多种 augmentation 结果取平均
        flips: 列表，可选 'none', 'hflip', 'vflip'
        """
        img_tensor, img_rgb = self.preprocess(image_path)
        
        all_probs = []
        with torch.no_grad():
            for flip in flips:
                if flip == 'none':
                    t = img_tensor
                elif flip == 'hflip':
                    t = torch.flip(img_tensor, dims=[3])
                elif flip == 'vflip':
                    t = torch.flip(img_tensor, dims=[2])
                else:
                    continue
                
                prob = self.model(t).cpu().numpy()
                
                # 如果是翻转的，翻转回来
                if flip == 'hflip':
                    prob = np.flip(prob, axis=3)
                elif flip == 'vflip':
                    prob = np.flip(prob, axis=2)
                
                all_probs.append(prob)
        
        # 平均所有 augmentation 的预测
        avg_prob = np.mean(all_probs, axis=0)
        
        # 取出预测结果 (模型输出已经是 input_size 尺寸)
        pred = avg_prob[0, 0]  # (H, W)
        
        # 调整到原始图像尺寸
        h, w = img_rgb.shape[:2]
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        
        pred_binary = (pred_resized > 0.5).astype(np.uint8)
        
        return img_rgb, pred_resized, pred_binary


def main():
    parser = argparse.ArgumentParser(description='VGA-Net 批量推理 (TTA)')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像文件夹')
    parser.add_argument('--mask', type=str, default=None, help='Ground Truth 文件夹')
    parser.add_argument('--output', type=str, default='../results_tta', help='结果保存文件夹')
    parser.add_argument('--tta', type=str, default='hflip', 
                        help="TTA模式: none(无TTA), hflip(水平翻转), hvflip(水平+垂直)")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.tta == 'none':
        flips = ['none']
    elif args.tta == 'hflip':
        flips = ['none', 'hflip']
    elif args.tta == 'hvflip':
        flips = ['none', 'hflip', 'vflip']
    else:
        flips = ['none', 'hflip']
    
    print(f"正在加载模型: {args.model}")
    print(f"TTA 模式: {flips}")
    inferencer = TTAInferencer(args.model)
    
    image_paths = sorted(glob.glob(os.path.join(args.input, '*.tif')))
    if not image_paths:
        print(f"错误: 在 {args.input} 中没有找到 .tif 文件")
        return
    
    print(f"找到 {len(image_paths)} 张图像，开始推理...")
    
    all_results = []
    for idx, img_path in enumerate(image_paths, 1):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx}/{len(image_paths)}] 处理: {basename}")
        
        img_rgb, pred_prob, pred_binary = inferencer.predict_tta(img_path, flips=flips)
        
        mask = None
        if args.mask and os.path.exists(args.mask):
            possible_masks = [
                os.path.join(args.mask, f"{basename}_manual1.gif"),
                os.path.join(args.mask, f"{basename.replace('test', 'manual1')}.gif"),
                os.path.join(args.mask, f"{basename}_mask.gif"),
            ]
            for mask_path in possible_masks:
                if os.path.exists(mask_path):
                    mask = inferencer.load_mask(mask_path, img_rgb.shape)
                    break
        
        metrics = compute_metrics(pred_binary, mask) if mask is not None else None
        
        pred_save = (pred_binary * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f"{basename}_pred.png"), pred_save)
        
        if mask is not None:
            h, w = img_rgb.shape[:2]
            comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
            comparison[:, :w] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            comparison[:, w:2*w] = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            comparison[:, 2*w:] = cv2.cvtColor(pred_save, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(args.output, f"{basename}_compare.png"), comparison)
        
        all_results.append({'name': basename, 'metrics': metrics})
    
    print("\n" + "="*70)
    print(f"推理结果汇总 (TTA: {args.tta})")
    print("="*70)
    print(f"{'文件名':<20} {'Accuracy':>10} {'Dice':>10} {'SE':>10} {'SP':>10} {'MCC':>10}")
    print("-"*70)
    
    for res in all_results:
        name = res['name']
        if res['metrics']:
            m = res['metrics']
            print(f"{name:<20} {m['Accuracy']:>10.4f} {m['Dice']:>10.4f} {m['SE']:>10.4f} {m['SP']:>10.4f} {m['MCC']:>10.4f}")
    
    valid_metrics = [r['metrics'] for r in all_results if r['metrics'] is not None]
    if valid_metrics:
        print("-"*70)
        avg = {
            'Accuracy': np.mean([m['Accuracy'] for m in valid_metrics]),
            'Dice': np.mean([m['Dice'] for m in valid_metrics]),
            'SE': np.mean([m['SE'] for m in valid_metrics]),
            'SP': np.mean([m['SP'] for m in valid_metrics]),
            'MCC': np.mean([m['MCC'] for m in valid_metrics]),
        }
        print(f"{'平均值':<20} {avg['Accuracy']:>10.4f} {avg['Dice']:>10.4f} {avg['SE']:>10.4f} {avg['SP']:>10.4f} {avg['MCC']:>10.4f}")
    
    print("="*70)
    print(f"结果已保存到: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
