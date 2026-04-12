# -*- coding: utf-8 -*-

import torch.nn as nn
from graph_construction import GraphConstruction
from pixel_feature_extraction import DRIU
from graph_feature_extraction import GraphFeatureExtraction
from segmentation import VGA_Net
from patch_extraction import extract_random_patches

class FinalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入图像尺寸
        self.input_size = (512, 512)

        # 图参数
        self.patch_size = 16
        self.hop_distance = 1

        # Dropout 比率
        self.dropout_rate = 0.5

        # **图构建部分**
        self.graph_construction = GraphConstruction(
            patch_size=self.patch_size, hop_distance=self.hop_distance
        )

        # **像素级特征提取部分**
        self.pixel_feature_extraction = DRIU(input_size=self.input_size)

        # **图级特征提取部分**
        self.graph_feature_extraction = GraphFeatureExtraction(
            dropout_rate=self.dropout_rate
        )

        # **分割部分**
        self.segmentation = VGA_Net()

    def forward(self, x):
        
        j = self.extract_random_patches(x)
        # **步骤 1：图构建**
        A = self.graph_construction(x)

        # **步骤 2：像素级特征提取**
        node_features = self.pixel_feature_extraction(x)

        # **步骤 3：图级特征提取**
        graph_features = self.graph_feature_extraction(A, node_features)

        # **步骤 4：分割**
        segmentation_output = self.segmentation(j,graph_features)

        return segmentation_output
