# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class HDCModule(nn.Module):
    """空洞卷积模块（加 BatchNorm 防止数值爆炸）"""
    def __init__(self):
        super().__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        features = torch.cat((path1_out, path2_out), dim=1)
        features = self.concat(features)
        return features
