# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from Model.graph_construction import GraphConstruction
from Model.pixel_feature_extraction import DRIU
from Model.graph_feature_extraction import GraphFeatureExtraction
from Model.segmentation import VGA_Net


class FinalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入图像尺寸（DRIVE数据集图像尺寸约为 584×565）
        self.input_size = (584, 565)

        # 图参数
        self.patch_size = 16
        self.hop_distance = 1

        # Dropout 比率
        self.dropout_rate = 0.5

        # 图特征提取参数
        # patch_size=16, RGB图像，所以每个图像块的特征维度是 3*16*16=768
        self.feature_dim = 3 * self.patch_size * self.patch_size
        self.num_heads = 2  # 必须设为2，与GATConv的chunk逻辑匹配
        self.num_layers = 2

        # **图构建部分**
        self.graph_construction = GraphConstruction(
            patch_size=self.patch_size, hop_distance=self.hop_distance
        )

        # **像素级特征提取部分**
        self.pixel_feature_extraction = DRIU(input_size=self.input_size)

        # **图级特征提取部分**
        self.graph_feature_extraction = GraphFeatureExtraction(
            dropout_rate=self.dropout_rate,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        # **分割部分**
        self.segmentation = VGA_Net(in_channels=2)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # **步骤 1：图构建**
        # A: (num_patches, num_patches) 邻接矩阵
        # patches: (batch, num_patches, feature_dim) 图像块特征
        A, patches = self.graph_construction(x)
        num_patches = patches.shape[1]
        
        # **步骤 2：像素级特征提取**
        # pixel_features: (batch, 1, H, W)
        pixel_features = self.pixel_feature_extraction(x)

        # **步骤 3：图级特征提取**
        # 对每个 batch 分别处理
        graph_features_list = []
        for i in range(batch_size):
            # patches[i]: (num_patches, feature_dim)
            gf = self.graph_feature_extraction(A, patches[i])
            graph_features_list.append(gf)
        
        # 合并 batch 的图特征: (batch, num_patches, feature_dim)
        graph_features = torch.stack(graph_features_list, dim=0)
        
        # 将图特征投影到与像素特征相同的维度 (1通道)
        # 简单平均池化所有图特征
        graph_features_mean = graph_features.mean(dim=-1, keepdim=True)  # (batch, num_patches, 1)
        
        # 将图特征 reshape 为空间格式
        # 计算空间布局 (graph_construction 已将输入 resize 到 512x512)
        h_patches = 512 // self.patch_size  # 32
        w_patches = 512 // self.patch_size  # 32
        
        graph_features_spatial = graph_features_mean.view(batch_size, h_patches, w_patches, 1)
        graph_features_spatial = graph_features_spatial.permute(0, 3, 1, 2)  # (batch, 1, h_patches, w_patches)
        
        # 上采样到 512x512
        graph_features_resized = nn.functional.interpolate(
            graph_features_spatial, 
            size=(512, 512), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 再调整到与 pixel_features 相同的尺寸 (584, 565)
        graph_features_resized = nn.functional.interpolate(
            graph_features_resized, 
            size=self.input_size, 
            mode='bilinear', 
            align_corners=False
        )

        # **步骤 4：分割**
        # 将 pixel 特征和 graph 特征拼接后传入 VGA-Net 进行精细分割
        # pixel_features: (batch, 1, H, W)
        # graph_features_resized: (batch, 1, H, W)
        combined_features = torch.cat([pixel_features, graph_features_resized], dim=1)  # (batch, 2, H, W)
        
        # VGA-Net：用 Attention Gate 替换 AB_FFMModule，图特征注入跳跃连接
        output = self.segmentation(combined_features, graph_features_resized)
        
        return output
