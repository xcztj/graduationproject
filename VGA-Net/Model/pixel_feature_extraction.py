# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DRIU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.num_channels = 16

        # 使用 ResNet-50 替代 VGG-16（更强的特征提取能力）
        self.base_network = ResNetBackbone()

        # 血管专用层：从 layer2 (512ch) 提取
        self.vessel_specialized_layers = nn.Sequential(
            nn.Conv2d(512, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 视盘专用层：从 layer3 (1024ch) 提取
        self.optic_disc_specialized_layers = nn.Sequential(
            nn.Conv2d(1024, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # 通过 ResNet-50 backbone
        feature_maps = self.base_network(x)

        # 提取像素级特征
        vessel_features = self.vessel_specialized_layers(feature_maps[0])
        optic_disc_features = self.optic_disc_specialized_layers(feature_maps[1])

        # 调整大小并组合特征
        vessel_features = F.interpolate(vessel_features, size=self.input_size, mode="bilinear")
        optic_disc_features = F.interpolate(optic_disc_features, size=self.input_size, mode="bilinear")
        combined_features = torch.cat((vessel_features, optic_disc_features), dim=1)

        # 返回 32 通道多尺度特征图
        return combined_features


class ResNetBackbone(nn.Module):
    """ResNet-50 Backbone（加载 ImageNet 预训练权重）
    
    输出：
    - features[0]: layer2 输出 (512ch, H/8, W/8) - 更多空间细节
    - features[1]: layer3 输出 (1024ch, H/16, W/16) - 更多语义信息
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 提取 ResNet-50 各层
        self.conv1 = resnet.conv1      # 7x7, stride=2, 64ch
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 3x3, stride=2
        
        self.layer1 = resnet.layer1    # 3 blocks, 256ch
        self.layer2 = resnet.layer2    # 4 blocks, 512ch, stride=2
        self.layer3 = resnet.layer3    # 6 blocks, 1024ch, stride=2
        # 不使用 layer4（太深，空间分辨率太小）

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        layer2_out = x  # (B, 512, H/8, W/8)
        
        x = self.layer3(x)
        layer3_out = x  # (B, 1024, H/16, W/16)
        
        return [layer2_out, layer3_out]
