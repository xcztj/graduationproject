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
        self.mask_files = sorted(os.listdir(os.path.join(root_dir, 'mask')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 根据 use_preprocessed 选择图像目录
        image_dir = 'preprocessed' if self.use_preprocessed else 'images'
        img_name = os.path.join(self.root_dir, image_dir, self.image_files[idx])
        mask_name = os.path.join(self.root_dir, 'mask', self.mask_files[idx])
        
        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        # 数据增强：随机翻转、旋转（仅训练集）
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            # 随机垂直翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            # 随机旋转 90/180/270 度（注意：旋转会交换宽高，需单独处理）
            # 暂时禁用旋转以避免 batch 内尺寸不一致
            # if random.random() > 0.5:
            #     k = random.choice([1, 2, 3])
            #     image = np.rot90(image, k).copy()
            #     mask = np.rot90(mask, k).copy()

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
