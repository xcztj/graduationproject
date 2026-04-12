# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AB_FFMModule(nn.Module):
    def __init__(self):
        super().__init__()

        # BConvLSTM

        self.bconvlstm = BConvLSTM(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        #  双重注意力网络

        self.attention = DualAttentionModule()

    def forward(self, x):
        # BConvLSTM

        features = self.bconvlstm(x)

        # 双重注意力网络

        features = self.attention(features)

        return features
class BConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        # 两个方向相反的 ConvLSTM

        self.forward_lstm = nn.ConvLSTM(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_lstm = nn.ConvLSTM(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # 通过两个 ConvLSTM

        forward_out, _ = self.forward_lstm(x)
        backward_out, _ = self.backward_lstm(x.flip(0))

        # 合并输出

        features = torch.cat((forward_out, backward_out), dim=1)

        return features



class DualAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 像素级注意力模块

        self.pam = PixelAttentionModule()

        # 通道级注意力模块

        self.cam = ChannelAttentionModule()

    def forward(self, x):
        # 像素级注意力

        pixel_weights = self.pam(x)

        # 通道级注意力

        channel_weights = self.cam(x)

        # 合并两种注意力

        features = x * pixel_weights * channel_weights

        return features



class PixelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层和最大池化

        self.conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 注意力图

        attention_map = self.conv(x)
        attention_map = self.pool(attention_map)

        # 像素级权重

        pixel_weights = torch.sigmoid(attention_map)

        return pixel_weights


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层和全局平均池化

        self.conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        # 注意力图

        attention_map = self.conv(x)
        attention_map = self.pool(attention_map)

        # 通道级权重

        channel_weights = torch.sigmoid(attention_map)

        return channel_weights
