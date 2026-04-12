# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DRIU(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # 输入图像尺寸
        self.input_size = input_size

        # 专用层的输出通道数
        self.num_channels = 16

        # 基础网络架构 (VGG)
        self.base_network = VGG(input_size=input_size)

        # 血管专用层
        self.vessel_specialized_layers = nn.Sequential(
            nn.Conv2d(512, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 视盘专用层
        self.optic_disc_specialized_layers = nn.Sequential(
            nn.Conv2d(512, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 特征融合的最终层
        self.final_layer = nn.Conv2d(self.num_channels * 2, 1, kernel_size=1)

    def forward(self, x):
        # 通过基础网络
        feature_maps = self.base_network(x)

        # 提取像素级特征
        vessel_features = self.vessel_specialized_layers(feature_maps[-4])
        optic_disc_features = self.optic_disc_specialized_layers(feature_maps[0])

        # 调整大小并组合特征
        vessel_features = F.interpolate(vessel_features, size=self.input_size, mode="bilinear")
        optic_disc_features = F.interpolate(optic_disc_features, size=self.input_size, mode="bilinear")
        combined_features = torch.cat((vessel_features, optic_disc_features), dim=1)

        # 预测
        segmentation_output = self.final_layer(combined_features)

        return segmentation_output

class VGG(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # VGG-16 架构
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x):
   
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        return [x]
