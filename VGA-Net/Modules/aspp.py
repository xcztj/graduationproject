# -*- coding: utf-8 -*-
"""
ASPP (Atrous Spatial Pyramid Pooling) 模块
DeepLabV3+ 风格的多尺度特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1×1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 3×3 conv, dilation=6
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 3×3 conv, dilation=12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 3×3 conv, dilation=18
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling + 1×1 conv（去掉 BN，避免 batch_size=1 时 1×1 特征图报错）
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 投影回 out_channels (5 个分支 concat 后的通道数 = out_channels // 4 * 5)
        concat_channels = (out_channels // 4) * 5
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(out)
