# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import ToTensor
from dataset import DRIVEDataset
from Model.VGA_Net import FinalNetwork
from Test.utils import train_model

# 定义 DRIVE 数据集目录路径（训练集）
drive_dataset_dir = '/root/autodl-tmp/VGA-Net/DRIVE/training'

# 定义转换
transform = ToTensor()

# 加载数据集（use_preprocessed=True 使用预处理后的图像）
drive_dataset = DRIVEDataset(root_dir=drive_dataset_dir, transform=transform, use_preprocessed=True)

# 将数据集拆分为训练集和测试集
train_size = int(0.7 * len(drive_dataset))
val_size = int(0.2 * len(drive_dataset))
test_size = len(drive_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(drive_dataset, [train_size, val_size, test_size])

# 定义数据加载器
# 注意：训练集只有14张图像，batch_size 不能大于数据集大小
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 定义模型、损失函数和优化器
model = FinalNetwork()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer)

# =============================================================================
# import os
# import cv2
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, 
# import numpy as np
# from torch.utils.data import DataLoader, random_split
# from torchvision.transforms import ToTensor
# from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import confusion_matrix
# 
# 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # 定义数据集类
# class DRIVEDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = sorted(os.listdir(os.path.join(root_dir, 'images')))
#         self.mask_files = sorted(os.listdir(os.path.join(root_dir, 'masks')))
# 
#     def __len__(self):
#         return len(self.image_files)
# 
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, 'images', self.image_files[idx])
#         mask_name = os.path.join(self.root_dir, 'masks', self.mask_files[idx])
#         
#         image = cv2.imread(img_name)
#         mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
# 
#         if self.transform:
#             sample = {'image': image, 'mask': mask}
#             sample = self.transform(sample)
# 
#         return sample
# 
# # 将图像转换为 PyTorch 张量
# class ToTensor(object):
#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
#         # 将图像从 BGR 格式转换为 RGB 格式并转换为张量
#         image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
#         # 将掩码转换为张量并从 (H, W) 调整为 (1, H, W)
#         mask = torch.from_numpy(mask).unsqueeze(0).float()
#         return {'image': image, 'mask': mask}
# 
# #DRIVE 数据集目录路径
# drive_dataset_dir = 'yeganeh/402/drive'
# 
# 
# # Load the dataset
# drive_dataset = DRIVEDataset(root_dir=drive_dataset_dir, transform=transform)
# 
# # Split the dataset into training and testing sets
# train_size = int(0.7 * len(drive_dataset))
# val_size = int(0.2 * len(drive_dataset))
# test_size = len(drive_dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = random_split(drive_dataset, [train_size, val_size, test_size])
# 
# # Define dataloaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)
# test_loader = DataLoader(test_dataset, batch_size=32)
# 
# # 定义模型架构
# class FinalNetwork(nn.Module):
#     def __init__(self):
#         super(FinalNetwork, self).__init__()
#         # 在此定义模型架构
# 
#     def forward(self, x):
#         # 定义模型的前向传播
#         return x
# 
# #训练
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     best_val_loss = float('inf')
#     early_stopping_counter = 0
# 
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for batch in train_loader:
#             inputs, labels = batch['image'].to(device), batch['mask'].to(device)
# 
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
# 
#             train_loss += loss.item() * inputs.size(0)
# 
#         train_loss /= len(train_loader.dataset)
# 
#         # Validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 inputs, labels = batch['image'].to(device), batch['mask'].to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * inputs.size(0)
# 
#         val_loss /= len(val_loader.dataset)
# 
#         # 早停
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stopping_counter = 0
#             torch.save(model.state_dict(), 'best_model.pt')  # 保存最佳模型
#         else:
#             early_stopping_counter += 1
#             if early_stopping_counter >= patience:
#                 print(f"No improvement in validation loss for {patience} epochs. Early stopping...")
#                 break
# 
#         print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
# 
# # Define the model, loss function, and optimizer
# model = FinalNetwork()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# 
# # Train the model
# train_model(model, train_loader, val_loader, criterion, optimizer)
# 
# # Load the best model
# model.load_state_dict(torch.load('best_model.pt'))
# def specificity_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp)
#     return specificity
# 
# def sensitivity_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     sensitivity = tp / (tp + fn)
#     return sensitivity
# 
# def dice_score(y_true, y_pred):
#     intersection = np.sum(y_true * y_pred)
#     dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
#     return dice
# 
# def centerline_dice_score(y_true, y_pred):
#     # Implement the calculation of centerline-Dice score
#     pass
# 
# def matthews_correlation_coefficient(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#     return mcc
# 
# 
# # 在测试集上评估模型
# model.eval()
# test_loss = 0.0
# predictions = []
# with torch.no_grad():
#     for batch in test_loader:
#         inputs, labels = batch['image'].to(device), batch['mask'].to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item() * inputs.size(0)
#         predictions.append(outputs.cpu().numpy())
# 
# test_loss /= len(test_loader.dataset)
# 
# # 计算评估指标
# predictions = np.concatenate(predictions, axis=0)
# test_labels = np.concatenate([batch['mask'].numpy() for batch in test_loader], axis=0)
# 
# ACC = accuracy_score(test_labels.flatten(), (predictions > 0.5).flatten())
# SP = specificity_score(test_labels.flatten(), (predictions > 0.5).flatten())
# SE = sensitivity_score(test_labels.flatten(), (predictions > 0.5).flatten())
# Dice = dice_score(test_labels.flatten(), (predictions > 0.5).flatten())
# clDice = centerline_dice_score(test_labels.flatten(), (predictions > 0.5).flatten())
# MCC = matthews_correlation_coefficient(test_labels.flatten(), (predictions > 0.5).flatten())
# 
# # 打印评估结果
# print("Test Loss:", test_loss)
# print("Matthews Correlation Coefficient:", MCC)
# print("Accuracy:", ACC)
# print("Specificity:", SP)
# print("Sensitivity:", SE)
# print("Dice Score:", Dice)
# print("Centerline Dice Score:", clDice)
# 
# =============================================================================
