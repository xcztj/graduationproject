# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """通道注意力模块 (Squeeze-and-Excitation)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HDCModule(nn.Module):
    """增强版空洞卷积模块：三条 dilation 路径 (2, 4, 8) + 残差 + SE"""
    def __init__(self):
        super().__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=8, dilation=8),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.concat = nn.Sequential(
            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128)
        )

        self.se = SEBlock(128, reduction=16)

    def forward(self, x):
        identity = self.shortcut(x)
        
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        
        features = torch.cat([p1, p2, p3], dim=1)
        features = self.concat(features)
        
        out = F.relu(features + identity)
        out = self.se(out)
        return out
