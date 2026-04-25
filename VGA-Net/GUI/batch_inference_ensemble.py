# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import cv2
import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True, help='模型路径列表')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--output', type=str, default='../results_ensemble')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 加载所有模型
    inferencers = []
    for model_path in args.models:
        print(f"加载模型: {model_path}")
        inferencers.append(VGAInferencer(model_path))
    
    image_paths = sorted(glob.glob(os.path.join(args.input, '*.tif')))
    print(f"找到 {len(image_paths)} 张图像，{len(inferencers)} 个模型集成推理中...")
    
    all_results = []
    for idx, img_path in enumerate(image_paths, 1):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx}/{len(image_paths)}] {basename}")
        
        # 收集所有模型的预测概率
        all_probs = []
        img_rgb = None
        for inf in inferencers:
            img_rgb, pred_prob, _ = inf.predict(img_path)
            all_probs.append(pred_prob)
        
        # 平均所有模型的概率
        avg_prob = np.mean(all_probs, axis=0)
        pred_binary = (avg_prob > 0.5).astype(np.uint8)
        
        mask = None
        if args.mask and os.path.exists(args.mask):
            possible_masks = [
                os.path.join(args.mask, f"{basename}_manual1.gif"),
                os.path.join(args.mask, f"{basename.replace('test', 'manual1')}.gif"),
                os.path.join(args.mask, f"{basename}_mask.gif"),
            ]
            for mask_path in possible_masks:
                if os.path.exists(mask_path):
                    mask = inferencers[0].load_mask(mask_path, img_rgb.shape)
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
    print(f"集成推理结果汇总 ({len(inferencers)} 个模型)")
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

if __name__ == '__main__':
    main()
