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

        # 注意力函数的权重
        self.W = torch.nn.Linear(feature_dim, feature_dim * num_heads, bias=False)
        
        # 可学习的偏置
        self.bias = torch.nn.Parameter(torch.zeros(1))

        # 注意力和最终输出的激活函数
        self.activation = F.leaky_relu

    def forward(self, inputs, adj):
        # Wh 和 Ws：节点特征（形状：[num_nodes, feature_dim/num_heads]）
        Wh, Ws = self.W(inputs).chunk(self.num_heads, dim=-1)
        
        # 计算注意力分数 (a) - 简化版本
        # 使用点积注意力
        a = torch.matmul(Wh, Ws.transpose(-1, -2))  # (num_nodes, num_nodes)
        a = F.softmax(a, dim=-1)

        # 应用邻接矩阵掩码
        a = a * adj
        
        # 特征的加权和
        out = torch.matmul(a, Wh)
        
        # 合并多头注意力结果
        out = out.view(out.shape[0], -1)  # (num_nodes, feature_dim)

        # 残差连接（需要投影到相同维度）
        if out.shape == inputs.shape:
            out = out + inputs
        
        out = self.activation(out)

        return out
