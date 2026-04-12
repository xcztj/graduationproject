# -*- coding: utf-8 -*-

import torch.nn as nn
import GCN

class GraphFeatureExtraction(nn.Module):
    def __init__(self, dropout_rate, feature_dim, num_heads, num_layers):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 用于图级特征提取的 GAT 模块
        self.graph_conv = GCN(feature_dim, num_heads, num_layers)

        # 用于防止过拟合的 Dropout 层
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, A, node_features):
        # 提取图级特征
        graph_features = self.graph_conv(node_features, A)

        # 应用 Dropout
        graph_features = self.dropout(graph_features)

        return graph_features
