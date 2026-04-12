# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM 单元，处理单个时间步"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLSTMCell, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 输入到状态的卷积：将输入和隐藏状态拼接后通过卷积
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,  # 4个门：i, f, g, o
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: 当前输入 (batch, in_channels, H, W)
            h_prev: 前一时刻隐藏状态 (batch, out_channels, H, W)
            c_prev: 前一时刻细胞状态 (batch, out_channels, H, W)
        Returns:
            h: 当前隐藏状态
            c: 当前细胞状态
        """
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=1)  # (batch, in_channels + out_channels, H, W)
        
        # 通过卷积得到4个门
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_g, cc_o = torch.split(conv_output, self.out_channels, dim=1)
        
        # 门控
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        g = torch.tanh(cc_g)     # 候选状态
        o = torch.sigmoid(cc_o)  # 输出门
        
        # 更新细胞状态和隐藏状态
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, c


class ConvLSTM(nn.Module):
    """ConvLSTM 网络，处理整个序列"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cell = ConvLSTMCell(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        Args:
            x: 输入序列 (seq_len, batch, in_channels, H, W) 或 (batch, in_channels, H, W)
        Returns:
            output: 输出序列 (seq_len, batch, out_channels, H, W)
            (h, c): 最后的隐藏状态和细胞状态
        """
        # 如果输入是4维的（无序列维度），添加序列维度
        if x.dim() == 4:
            x = x.unsqueeze(0)  # (1, batch, in_channels, H, W)
        
        seq_len, batch, _, H, W = x.size()
        
        # 初始化隐藏状态和细胞状态
        h = torch.zeros(batch, self.out_channels, H, W, device=x.device)
        c = torch.zeros(batch, self.out_channels, H, W, device=x.device)
        
        # 存储输出
        outputs = []
        
        # 遍历序列
        for t in range(seq_len):
            h, c = self.cell(x[t], h, c)
            outputs.append(h)
        
        # 拼接输出
        output = torch.stack(outputs, dim=0)  # (seq_len, batch, out_channels, H, W)
        
        return output, (h, c)


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

        self.forward_lstm = ConvLSTM(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_lstm = ConvLSTM(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

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
