# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # 裁剪到 (1e-7, 1-1e-7) 防止 BCELoss 在预测接近 0/1 时爆炸
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity

def dice_score(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    dice = (2.0 * intersection) / (torch.sum(y_true) + torch.sum(y_pred))
    return dice

def centerline_dice_score(y_true, y_pred):
    # 实现 centerline-Dice 分数的计算
    pass

def matthews_correlation_coefficient(y_true, y_pred):
    import numpy as np
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 防止溢出，转换为 float64
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    mcc = numerator / denominator
    return mcc


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch['image'].to(device), batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['image'].to(device), batch['mask'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), '/root/autodl-tmp/VGA-Net/Train/best_model.pt')  # 保存最佳模型
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"验证损失连续 {patience} 轮没有改善，早停...")
                break

        print(f"轮次 [{epoch+1}/{num_epochs}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
