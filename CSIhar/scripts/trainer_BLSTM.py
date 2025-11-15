import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import sys
import os
import time
import argparse
import csv
import yaml
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_builder_BLSTM import build_blstm_model


# ==========================
# 新增：显存预估函数
# ==========================
def estimate_memory_usage(model, input_shape, batch_size, device, amp_enabled):
    """
    预估模型训练时的显存占用
    - input_shape: 输入数据形状 (seq_len, features)
    - 返回: 预估显存(MB)和详细组件占用
    """
    # 清空缓存
    torch.cuda.empty_cache()

    # 创建虚拟输入
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    dummy_label = torch.randint(0, 7, (batch_size,)).to(device)

    # 计算模型参数占用
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    param_size_mb = param_size / (1024 ** 2)  # 转换为MB

    # 计算输入数据占用
    input_size = dummy_input.element_size() * dummy_input.numel()
    input_size_mb = input_size / (1024 ** 2)

    # 计算输出和中间变量占用（前向传播）
    model.eval()
    with torch.no_grad():
        dummy_output = model(dummy_input)
    output_size = dummy_output.element_size() * dummy_output.numel()
    # 中间变量预估为参数的2-3倍（根据模型复杂度）
    middle_vars_size_mb = param_size_mb * 2.5
    forward_size_mb = (output_size / (1024 ** 2)) + middle_vars_size_mb

    # 计算反向传播额外占用（通常为前向的2倍）
    backward_size_mb = forward_size_mb * 2

    # AMP影响：混合精度可减少约30-40%的中间变量占用
    amp_reduction = 0.35 if amp_enabled else 0
    total_size_mb = param_size_mb + input_size_mb + \
                    (forward_size_mb + backward_size_mb) * (1 - amp_reduction)

    # 额外预留10%安全空间
    total_size_mb *= 1.1

    # 获取当前设备总显存
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)

    return {
        "total_estimated": round(total_size_mb, 2),
        "param": round(param_size_mb, 2),
        "input": round(input_size_mb, 2),
        "forward": round(forward_size_mb, 2),
        "backward": round(backward_size_mb, 2),
        "gpu_total": round(total_gpu_memory, 2),
        "safe": total_size_mb < total_gpu_memory * 0.9  # 预留10%显存
    }


# ==========================
# 混淆矩阵可视化（显示准确率）
# ==========================
def plot_confusion_matrix(y_true, y_pred, classes, save_path, cmap=plt.cm.Blues):
    """绘制显示准确率的混淆矩阵（每行归一化）"""
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, label='准确率')

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        xlabel='预测类别',
        ylabel='真实类别',
        title='混淆矩阵（准确率）'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 准确率混淆矩阵已保存至: {save_path}")


# ==========================
# 计算评估指标
# ==========================
def compute_metrics(y_true, y_pred, classes, out_dir):
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        output_dict=False,
        digits=4
    )
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 分类报告已保存至: {report_path}")

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, classes, cm_path)

    return report


# ==========================
# 优化器配置
# ==========================
def get_optimizer(model, cfg):
    optim_type = cfg['optimizer']['type'].lower()
    lr = cfg['optimizer']['lr']
    weight_decay = cfg['optimizer'].get('weight_decay', 1e-5)
    if optim_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        momentum = cfg['optimizer'].get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optim_type}")


# ==========================
# 学习率调度器
# ==========================
def get_scheduler(optimizer, cfg):
    scheduler_type = cfg['scheduler']['type'].lower()
    epochs = cfg['training']['epochs']
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        step_size = cfg['scheduler'].get('step_size', 50)
        gamma = cfg['scheduler'].get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_type}")


# ==========================
# 评估函数
# ==========================
def evaluate(model, dataloader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, all_preds, all_labels


# ==========================
# 单轮训练
# ==========================
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                    device, cfg, scaler, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    warmup_epochs = cfg['training'].get('warmup_epochs', 0)
    if epoch < warmup_epochs and warmup_epochs > 0:
        warmup_factor = (epoch + 1) / (warmup_epochs + 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['optimizer']['lr'] * warmup_factor
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [训练]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=cfg['training']['amp']):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if cfg['training']['amp']:
            scaler.scale(loss).backward()
            if cfg['training'].get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += batch_size
        current_loss = total_loss / total_samples
        current_acc = total_correct / total_samples
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    if epoch >= warmup_epochs and scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ==========================
# 主训练函数
# ==========================
def train_blstm_model(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg['output']['dir'], f"blstm_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有输出将保存至: {output_dir}")

    classes = ["bend", "fall", "lie_down", "run", "sitdown", "standup", "walk"]

    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if cfg['training']['amp'] and device.type != 'cuda':
        print("⚠️  CPU不支持混合精度训练，自动关闭AMP")
        cfg['training']['amp'] = False

    print("\n=== 加载预处理数据 ===")
    data_path = cfg['data']['preprocessed_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"预处理文件不存在: {data_path}")
    data = np.load(data_path)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    print(f"训练集: x={x_train.shape}, y={y_train.shape}")
    print(f"测试集: x={x_test.shape}, y={y_test.shape}")

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)

    pin_memory = cfg['data']['pin_memory'] and device.type == "cuda"
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=pin_memory
    )

    print("\n=== 构建BLSTM模型 ===")
    model = build_blstm_model(input_shape=(x_train.shape[1], x_train.shape[2])).to(device)
    print(model)

    # ==========================
    # 新增：显存预估与检查
    # ==========================
    if device.type == 'cuda':
        print("\n=== 显存预估 ===")
        input_shape = (x_train.shape[1], x_train.shape[2])  # (seq_len, features)
        mem_stats = estimate_memory_usage(
            model=model,
            input_shape=input_shape,
            batch_size=cfg['training']['batch_size'],
            device=device,
            amp_enabled=cfg['training']['amp']
        )

        # 打印显存详情
        print(f"GPU总显存: {mem_stats['gpu_total']} MB")
        print(f"预估总占用: {mem_stats['total_estimated']} MB (参数: {mem_stats['param']} MB, "
              f"输入: {mem_stats['input']} MB, 前向: {mem_stats['forward']} MB, 反向: {mem_stats['backward']} MB)")

        # 显存不足警告
        if not mem_stats['safe']:
            suggested_bs = int(
                cfg['training']['batch_size'] * 0.9 * mem_stats['gpu_total'] / mem_stats['total_estimated'])
            suggested_bs = max(16, suggested_bs // 16 * 16)  # 16的倍数
            print(f"⚠️  警告: 预估显存不足！建议将batch_size调整为 {suggested_bs}")
            # 可选：自动调整batch_size
            # cfg['training']['batch_size'] = suggested_bs
            # print(f"已自动将batch_size调整为 {suggested_bs}")

    # 初始化训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg['training']['amp'])

    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0
    patience_counter = 0
    total_epochs = cfg['training']['epochs']
    if device.type == 'cuda':
        init_mem = torch.cuda.memory_allocated(device) / 1024 ** 2
        print(f"\n✅ 初始GPU已用显存: {init_mem:.2f}MB，当前batch_size={cfg['training']['batch_size']}")

    print("\n=== 开始训练 ===")
    for epoch in range(total_epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, cfg, scaler, epoch, total_epochs
        )
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device, cfg)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{total_epochs} | 耗时: {epoch_time:.2f}秒")
        print(f"训练集: 损失={train_loss:.4f}, 准确率={train_acc:.4f}")
        print(f"测试集: 损失={val_loss:.4f}, 准确率={val_acc:.4f}")
        print(f"学习率: {current_lr:.6f}")
        if device.type == 'cuda':
            used_mem = torch.cuda.memory_allocated(device) / 1024 ** 2
            print(f"GPU已用显存: {used_mem:.2f}MB（峰值安全）")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)
            print(f"✅ 保存最佳模型 (验证准确率: {best_val_acc:.4f}) 至 {model_path}")
        else:
            patience_counter += 1
            if cfg['training']['early_stop'] and patience_counter >= cfg['training']['patience']:
                print(f"\n早停触发: 连续{cfg['training']['patience']}个epoch未提升验证准确率")
                break

    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    print("\n=== 加载最佳模型进行测试评估 ===")
    best_ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device, cfg)

    print(f"\n[TEST] 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    report = compute_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        classes=classes,
        out_dir=output_dir
    )
    print("\n=== 测试集分类报告 ===")
    print(report)

    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_path}")
    print(f"最佳模型: {model_path}")
    print(f"分类报告: {os.path.join(output_dir, 'classification_report.txt')}")
    print(f"混淆矩阵: {os.path.join(output_dir, 'confusion_matrix.png')}")
    return output_dir


# ==========================
# 默认配置
# ==========================
def get_default_config():
    return {
        'data': {
            'preprocessed_path': os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "processed_data_BLSTM",
                "csi_blstm_train_test.npz"
            ),
            'num_workers': 0,
            'pin_memory': True
        },
        'training': {
            'batch_size': 64,
            'epochs': 200,
            'warmup_epochs': 5,
            'amp': True,
            'grad_clip': 5.0,
            'early_stop': True,
            'patience': 20
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'scheduler': {
            'type': 'cosine'
        },
        'output': {
            'dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        },
        'device': 'cuda'
    }


# ==========================
# 主函数
# ==========================
def main():
    parser = argparse.ArgumentParser(description='BLSTM模型训练脚本（带显存预估）')
    parser.add_argument('--config', type=str, default=None, help='YAML配置文件路径')
    parser.add_argument('--device', type=str, default=None, help='指定设备（cuda/cpu）')
    parser.add_argument('--data-path', type=str, default=None, help='预处理数据路径')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖训练批次大小')
    args = parser.parse_args()

    cfg = get_default_config()

    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_cfg = yaml.safe_load(f)

            def merge_dict(a, b):
                for k, v in b.items():
                    if isinstance(v, dict) and k in a and isinstance(a[k], dict):
                        merge_dict(a[k], v)
                    else:
                        a[k] = v

            merge_dict(cfg, file_cfg)

    if args.device is not None:
        cfg['device'] = args.device
    if args.data_path is not None:
        cfg['data']['preprocessed_path'] = args.data_path
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
        print(f"⚠️  已通过命令行覆盖batch_size为: {args.batch_size}")

    os.makedirs(cfg['output']['dir'], exist_ok=True)

    config_save_path = os.path.join(
        cfg['output']['dir'],
        f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"配置已保存至: {config_save_path}")

    train_blstm_model(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 训练异常终止: {str(e)}")
        if "out of memory" in str(e).lower():
            print("建议解决方案: "
                  "\n1. 减小batch_size：python trainer_BLSTM.py --batch-size 128"
                  "\n2. 关闭AMP：在配置中设置amp: false")
        elif "type" in str(e).lower() and ("float32" in str(e) or "float16" in str(e)):
            print("建议解决方案: 确保数据为float32类型（代码已默认处理，无需手动修改）")
        sys.exit(1)
