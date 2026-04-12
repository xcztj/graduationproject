# -*- coding: utf-8 -*-


import os
import cv2
import torch
from torch.utils.data import Dataset


class DRIVEDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_preprocessed=False):
        """
        Args:
            root_dir: 数据集根目录
            transform: 数据转换
            use_preprocessed: 是否使用预处理后的图像（默认为False使用原始images目录）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_preprocessed = use_preprocessed
        
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
