# -*- coding: utf-8 -*-
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


class MultiScaleInferencer(VGAInferencer):
    def predict_multiscale(self, image_path, scales=[0.9, 1.0, 1.1]):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"无法读取图像: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        all_probs = []
        with torch.no_grad():
            for scale in scales:
                if scale == 1.0:
                    img_resized = img_rgb
                else:
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_resized = cv2.resize(img_rgb, (new_w, new_h))
                
                # 预处理
                img_tensor = cv2.resize(img_resized, self.input_size[::-1])
                img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                prob = self.model(img_tensor).cpu().numpy()[0, 0]
                
                # resize 回原始尺寸
                prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                all_probs.append(prob)
        
        avg_prob = np.mean(all_probs, axis=0)
        pred_binary = (avg_prob > 0.5).astype(np.uint8)
        return img_rgb, avg_prob, pred_binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--output', type=str, default='../results_multiscale')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    inferencer = MultiScaleInferencer(args.model)
    image_paths = sorted(glob.glob(os.path.join(args.input, '*.tif')))
    
    print(f"找到 {len(image_paths)} 张图像，多尺度推理中...")
    all_results = []
    for idx, img_path in enumerate(image_paths, 1):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx}/{len(image_paths)}] {basename}")
        
        img_rgb, pred_prob, pred_binary = inferencer.predict_multiscale(img_path)
        
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
    print("多尺度推理结果汇总")
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
