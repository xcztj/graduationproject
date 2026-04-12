# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numy as np
from torch.utils.data import DataLoader
import transforms
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
from dataset import DRIVEDataset
from model import FinalNetwork
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义 DRIVE 数据集目录路径
drive_dataset_dir = 'path/to/drive/dataset'

transforms = transforms.Compose([
    ToTensor(),
    # 如需其他转换请添加
])

# 加载数据集
drive_dataset = DRIVEDataset(root_dir=drive_dataset_dir, transforms=None)

# 为测试集定义数据加载器
test_loader = DataLoader(drive_dataset, batch_size=32)

# 定义模型
model = FinalNetwork()

criterion = nn.CrossEntropyLoss()

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pt'))

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

ACC = accuracy_score(test_labels.flatten(), (predictions > 0.5).flatten())
SP = utils.specificity_score(test_labels.flatten(), (predictions > 0.5).flatten())
SE = utils.sensitivity_score(test_labels.flatten(), (predictions > 0.5).flatten())
Dice = utils.dice_score(test_labels.flatten(), (predictions > 0.5).flatten())
clDice = utils.centerline_dice_score(test_labels.flatten(), (predictions > 0.5).flatten())
MCC = utils.matthews_correlation_coefficient(test_labels.flatten(), (predictions > 0.5).flatten())

# 打印评估结果
print("Test Loss:", test_loss)
print("Matthews Correlation Coefficient:", MCC)
print("Accuracy:", ACC)
print("Specificity:", SP)
print("Sensitivity:", SE)
print("Dice Score:", Dice)
print("Centerline Dice Score:", clDice)
