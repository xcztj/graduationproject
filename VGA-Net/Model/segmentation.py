# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.hdc_module import HDCModule
from Modules.attention_gate import AttentionGate

class VGA_Net(nn.Module):
    """VGA-Net 分割网络：3 层 U-Net + Attention Gate + Deep Supervision"""
    def __init__(self, in_channels=33):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        self.graph_proj = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        
        self.residual_proj = nn.Conv2d(in_channels, 1, kernel_size=1)

        # 3 层 Encoder
        self.encoder1 = HDCModule()
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = HDCModule()
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = HDCModule()
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = HDCModule()

        # 3 层 Decoder
        self.decoder3 = HDCModule()
        self.decoder2 = HDCModule()
        self.decoder1 = HDCModule()

        # Attention Gates
        self.att_gate1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att_gate2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att_gate3 = AttentionGate(F_g=128, F_l=128, F_int=64)

        # Deep Supervision outputs
        self.out_conv1 = nn.Conv2d(128, 1, kernel_size=1)
        self.out_conv2 = nn.Conv2d(128, 1, kernel_size=1)
        self.out_conv3 = nn.Conv2d(128, 1, kernel_size=1)
        self.output = nn.Sigmoid()

    def forward(self, x, graph_features=None):
        residual = self.residual_proj(x)
        orig_size = residual.shape[2:]

        x = self.input_proj(x)

        # Encoder
        e1 = self.encoder1(x)
        e1_p = self.pool1(e1)
        e2 = self.encoder2(e1_p)
        e2_p = self.pool2(e2)
        e3 = self.encoder3(e2_p)
        e3_p = self.pool3(e3)
        b = self.bottleneck(e3_p)

        # Decoder
        d3 = self.decoder3(b)
        d3_u = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(d3_u)
        d2_u = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(d2_u)
        d1_u = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)

        # Graph feature injection
        if graph_features is not None:
            gp = self.graph_proj(graph_features)
            e1 = e1 + gp
            gp2 = F.interpolate(gp, scale_factor=0.5, mode='bilinear', align_corners=False)
            e2 = e2 + gp2
            gp4 = F.interpolate(gp, scale_factor=0.25, mode='bilinear', align_corners=False)
            e3 = e3 + gp4

        # Attention Gates
        e3_att = self.att_gate3(g=d3_u, x=e3)
        e2_att = self.att_gate2(g=d2_u, x=e2)
        e1_att = self.att_gate1(g=d1_u, x=e1)

        # Upsample all to original size
        e3_att_u = F.interpolate(e3_att, size=orig_size, mode='bilinear', align_corners=False)
        e2_att_u = F.interpolate(e2_att, size=orig_size, mode='bilinear', align_corners=False)

        # Fusion
        fused = d1_u + e1_att + e2_att_u + e3_att_u
        fused = fused / 4.0

        # Deep supervision outputs
        out1 = self.out_conv1(fused)
        
        d2_u_full = F.interpolate(d2_u, size=orig_size, mode='bilinear', align_corners=False)
        out2 = self.out_conv2(d2_u_full)
        
        d3_u_full = F.interpolate(d3_u, size=orig_size, mode='bilinear', align_corners=False)
        out3 = self.out_conv3(d3_u_full)

        # Weighted sum
        final = 0.5 * out1 + 0.3 * out2 + 0.2 * out3
        return self.output(final + residual)
