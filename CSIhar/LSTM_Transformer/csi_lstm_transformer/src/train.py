import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# 新增：导入utils中的报告和绘图函数
from .utils import generate_classification_report, plot_confusion_matrix


def train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, epoch, warmup_steps):
    """训练单个epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for step, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)

        # 学习率预热
        if epoch < warmup_steps:
            current_step = epoch * len(train_loader) + step
            lr = 0.001 * current_step / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(data)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 统计指标
        total_loss += loss.item() * data.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += data.size(0)

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples * 100:.2f}%"
        )

    return total_loss / total_samples, total_correct / total_samples * 100


@torch.no_grad()
def evaluate(model, loader, device, criterion, classes=None, save_path=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * data.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += data.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100

    # 生成分类报告和混淆矩阵（如果传入classes和save_path）
    report = None
    if classes is not None and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # 生成分类报告
        report = generate_classification_report(all_labels, all_preds, classes, save_path)
        # 绘制混淆矩阵
        plot_confusion_matrix(all_labels, all_preds, classes, save_path)

    return avg_loss, avg_acc, all_preds, all_labels, report
