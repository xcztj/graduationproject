# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.hdc_module import HDCModule
from Modules.attention_gate import AttentionGate

class VGA_Net(nn.Module):
    """VGA-Net 分割网络：保留 HDCModule，用 Attention Gate 替换 AB_FFMModule"""
    def __init__(self, in_channels=2):
        super().__init__()

        # 输入投影：将 pixel_features + graph_features 映射到 128 通道
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        # 图特征投影：将 1 通道图特征映射到 128 通道（用于注入跳跃连接）
        self.graph_proj = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        # 编码器（独立权重）
        self.encoder1 = HDCModule()
        self.encoder2 = HDCModule()
        self.bottleneck = HDCModule()

        # 解码器（独立权重）
        self.decoder1 = HDCModule()
        self.decoder2 = HDCModule()

        # 注意力门控模块（核心创新：替换原 AB_FFMModule）
        self.att_gate1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att_gate2 = AttentionGate(F_g=128, F_l=128, F_int=64)

        # 输出层
        self.out_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.output = nn.Sigmoid()

    def forward(self, x, graph_features=None):
        """
        Args:
            x: 融合特征 (batch, in_channels, H, W)，默认是 pixel + graph 拼接
            graph_features: 图特征 (batch, 1, H, W)，用于注入跳跃连接
        """
        # 输入投影到 128 通道
        x = self.input_proj(x)  # (batch, 128, H, W)

        # 编码器路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        b = self.bottleneck(e2)

        # 解码器路径
        d2 = self.decoder1(b)
        d1 = self.decoder2(d2)

        # 投影图特征并注入跳跃连接（忠实原论文设计）
        if graph_features is not None:
            graph_proj = self.graph_proj(graph_features)
            e1 = e1 + graph_proj
            e2 = e2 + graph_proj

        # 注意力门控跳跃连接：用解码器特征门控编码器特征
        e2_att = self.att_gate2(g=d2, x=e2)
        e1_att = self.att_gate1(g=d1, x=e1)

        # 融合解码器输出与注意力门控后的跳跃连接
        x = d1 + e1_att + e2_att
        x = x / 3.0  # 缩放防止数值过大

        # 输出层
        x = self.out_conv(x)
        x = self.output(x)

        return x
