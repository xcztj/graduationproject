# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F
from Modules.hdc_module import HDCModule
from Modules.ab_ffm_module import AB_FFMModule
from Modules.GCN import GATConv 
from Modules.dropblock_module import Dropblock

class VGA_Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # HDC 模块
        self.hdc = HDCModule()

        # AB-FFM 模块
        self.ab_ffm = AB_FFMModule()

        # Dropblock 方法
        self.dropblock = Dropblock(block_size=7, drop_prob=0.15)

        # 编码器路径
        self.encoder1 = self.hdc
        self.encoder2 = self.hdc
        self.encoder3 = self.hdc
        self.encoder4 = self.hdc

        # 解码器路径
        self.decoder1 = self.hdc
        self.decoder2 = self.hdc
        self.decoder3 = self.hdc
        self.decoder4 = self.hdc

        # 输出层
        self.output = nn.Sigmoid()

        # 图注意力模块
        self.gat = GATConv(feature_dim=512, num_heads=8)

    def forward(self, patches, A):
        # 编码器路径
        x1 = self.encoder1(patches)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # HDC 模块
        x4 = self.hdc(x4)

        # 解码器路径
        x3 = self.decoder1(x4)
        x2 = self.decoder2(x3)
        x1 = self.decoder3(x2)
        x = self.decoder4(x1)

        # 图注意力模块
        A_out = self.gat(x, A)

        # 调整 A_out 的大小以匹配特征图尺寸
        A_out = F.interpolate(A_out, size=x.size()[2:], mode='bilinear', align_corners=True)

        # 将 A_out 注入每个跳跃连接
        x1 = x1 + A_out
        x2 = x2 + A_out
        x3 = x3 + A_out

        # AB-FFM 模块
        x = self.ab_ffm(x, x1, x2, x3)

        # Dropblock 方法
        x = self.dropblock(x)

        # 输出层
        x = self.output(x)

        return x
