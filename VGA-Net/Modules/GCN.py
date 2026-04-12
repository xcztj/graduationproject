# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers):
        super(GCN, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # GAT 层
        self.gat_layers = torch.nn.ModuleList([GATConv(feature_dim, num_heads) for _ in range(num_layers)])

    def forward(self, inputs, adj):
        # 按顺序应用 GAT 层
        for gat_layer in self.gat_layers:
            inputs = gat_layer(inputs, adj)

        # GCN 输出
        return inputs

class GATConv(torch.nn.Module):
    def __init__(self, feature_dim, num_heads, **kwargs):
        super(GATConv, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # 注意力函数的权重和偏置
        self.W = torch.nn.Linear(feature_dim, feature_dim * num_heads, bias=False)
        self.a = torch.nn.Linear(1, 1, bias=True)

        # 注意力和最终输出的激活函数
        self.activation = F.leaky_relu

    def forward(self, inputs, adj):
        # Wh 和 Ws：节点特征（形状：[num_heads, feature_dim]）
        Wh, Ws = self.W(inputs).chunk(self.num_heads, dim=-1)
        bias = self.a(torch.ones_like(Wh[:, :1])).view(-1)

        # 计算注意力分数 (a)
        a = torch.matmul(torch.tanh(Wh + Ws + bias), Wh.transpose(-1, -2))
        a = F.softmax(a, dim=-1)

        # 特征的加权和
        out = torch.matmul(a, Wh)

        # 残差连接和层归一化
        out = out + inputs
        out = F.batch_norm(out, training=self.training)
        out = self.activation(out)

        return out
