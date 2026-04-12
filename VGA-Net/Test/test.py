# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from Train.dataset import DRIVEDataset, ToTensor
from Model.VGA_Net import FinalNetwork
import Test.utils as utils

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义 DRIVE 数据集目录路径
drive_dataset_dir = '/root/autodl-tmp/VGA-Net/DRIVE/test'

# 加载数据集
drive_dataset = DRIVEDataset(root_dir=drive_dataset_dir, transform=ToTensor(), use_preprocessed=True)

# 为测试集定义数据加载器
test_loader = DataLoader(drive_dataset, batch_size=4)

# 定义模型
model = FinalNetwork()
model.to(device)

criterion = nn.BCELoss()

# 加载最佳模型
model.load_state_dict(torch.load('/root/autodl-tmp/VGA-Net/Train/best_model.pt', map_location=device))

# 在测试集上评估模型
model.eval()
test_loss = 0.0
predictions = []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch['image'].to(device), batch['mask'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        predictions.append(outputs.cpu().numpy())

test_loss /= len(test_loader.dataset)

# 计算评估指标
predictions = np.concatenate(predictions, axis=0)
test_labels = np.concatenate([batch['mask'].numpy() for batch in test_loader], axis=0)

# 转换为 PyTorch Tensor 以兼容 utils 函数
pred_tensor = torch.from_numpy((predictions > 0.5).astype(np.float32))
label_tensor = torch.from_numpy(test_labels.astype(np.float32))

ACC = accuracy_score(test_labels.flatten(), (predictions > 0.5).flatten())
SP = utils.specificity_score(test_labels.flatten(), (predictions > 0.5).flatten())
SE = utils.sensitivity_score(test_labels.flatten(), (predictions > 0.5).flatten())
Dice = utils.dice_score(label_tensor.flatten(), pred_tensor.flatten())
clDice = utils.centerline_dice_score(test_labels.flatten(), (predictions > 0.5).flatten())
MCC = utils.matthews_correlation_coefficient(label_tensor.flatten().numpy(), pred_tensor.flatten().numpy())

# 打印评估结果
print("Test Loss:", test_loss)
print("Matthews Correlation Coefficient:", MCC)
print("Accuracy:", ACC)
print("Specificity:", SP)
print("Sensitivity:", SE)
print("Dice Score:", Dice)
print("Centerline Dice Score:", clDice)
