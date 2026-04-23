# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """注意力门控模块 (Attention Gate)
    
    用于 U-Net 跳跃连接，根据解码器特征（gating signal）
    对编码器特征进行加权，抑制不相关区域，增强目标区域响应。
    
    参考: Attention U-Net: Learning Where to Look for the Pancreas (Oktay et al., 2018)
    
    Args:
        F_g: 门控信号（解码器特征）的通道数
        F_l: 跳跃连接（编码器特征）的通道数
        F_int: 中间特征通道数
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: 门控信号，来自解码器较粗尺度的特征 (B, F_g, H_g, W_g)
            x: 跳跃连接特征，来自编码器较细尺度的特征 (B, F_l, H_x, W_x)
        Returns:
            注意力加权后的跳跃连接特征 (B, F_l, H_x, W_x)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
