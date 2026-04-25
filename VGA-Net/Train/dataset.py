# -*- coding: utf-8 -*-


import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class DRIVEDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_preprocessed=False, augment=False):
        """
        Args:
            root_dir: 数据集根目录
            transform: 数据转换
            use_preprocessed: 是否使用预处理后的图像（默认为False使用原始images目录）
            augment: 是否对训练数据进行随机增强（仅训练集使用）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_preprocessed = use_preprocessed
        self.augment = augment
        
        # 根据 use_preprocessed 选择图像目录
        image_dir = 'preprocessed' if use_preprocessed else 'images'
        # 只加载 .tif 格式的文件，确保与 mask 文件数量一致
        self.image_files = sorted([f for f in os.listdir(os.path.join(root_dir, image_dir)) if f.endswith('.tif')])
        # 使用 1st_manual 作为血管标注 ground truth（而非 mask/ 下的 FOV mask）
        self.mask_files = sorted([f for f in os.listdir(os.path.join(root_dir, '1st_manual')) if f.endswith('.gif')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 根据 use_preprocessed 选择图像目录
        image_dir = 'preprocessed' if self.use_preprocessed else 'images'
        img_name = os.path.join(self.root_dir, image_dir, self.image_files[idx])
        mask_name = os.path.join(self.root_dir, '1st_manual', self.mask_files[idx])
        
        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        # 数据增强：随机翻转、缩放、亮度（仅训练集）
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            # 随机垂直翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            # 随机缩放 0.9-1.1
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h))
                if scale > 1.0:
                    start_y = (new_h - h) // 2
                    start_x = (new_w - w) // 2
                    image = image[start_y:start_y+h, start_x:start_x+w]
                    mask = mask[start_y:start_y+h, start_x:start_x+w]
                else:
                    pad_y = (h - new_h) // 2
                    pad_x = (w - new_w) // 2
                    image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_CONSTANT, value=0)
                    mask = cv2.copyMakeBorder(mask, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_CONSTANT, value=0)
            # 随机亮度调整
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)

        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # 确保转换为 float 类型并归一化到 [0, 1]
        image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous().float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous().float() / 255.0
        return {'image': image, 'mask': mask}
