# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class HDCModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 两个具有不同空洞率的路径
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.ReLU(),
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=4),
            nn.ReLU(),
        )

        # 拼接两个路径
        self.concat = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 通过两个路径
        path1_out = self.path1(x)
        path2_out = self.path2(x)

        # 拼接两个路径
        features = torch.cat((path1_out, path2_out), dim=1)
        features = self.concat(features)

        return features
