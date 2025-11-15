# 1. 导入语句移至文件开头（规范且避免执行顺序问题）
import argparse
import yaml
import time
import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.seed import set_seed
from csi_dataset import collect_files, CSIDataset, load_csi_file, haar_wavelet_decompose
from utils.transforms import CSIPseudoColorConverter
from models.csi_cnn import build_model
from utils.metrics import compute_metrics, accuracy_topk


def _to_float(name, val, default):
    if val is None:
        print(f"[WARN] {name} 未设置，采用默认 {default}")
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip())
    except Exception:
        print(f"[WARN] {name}={val} 不能转换为 float，采用默认 {default}")
        return default


def parse_lr(lr_cfg):
    """解析学习率配置，支持多种格式"""
    if isinstance(lr_cfg, (list, tuple)):
        if len(lr_cfg) == 0:
            print("[WARN] lr 列表为空，回退 1e-3")
            return 1e-3, 1e-3
        vals = []
        for v in lr_cfg:
            try:
                vals.append(float(str(v).strip()))
            except Exception:
                pass
        if len(vals) == 0:
            print("[WARN] lr 列表元素无法解析，回退 1e-3")
            return 1e-3, 1e-3
        if len(vals) == 1:
            return vals[0], vals[0]
        return vals[0], vals[-1]
    else:
        try:
            v = float(str(lr_cfg).strip())
            return v, v
        except Exception:
            print(f"[WARN] lr={lr_cfg} 解析失败，回退 1e-3")
            return 1e-3, 1e-3


def get_optimizer(parameters, cfg):
    tr = cfg['training']
    opt_name = str(tr.get('optimizer', 'adamw')).lower()
    lr = tr['main_base_lr']
    wd = _to_float('weight_decay', tr.get('weight_decay', 1e-4), 1e-4)
    if opt_name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        momentum = _to_float('momentum', tr.get('momentum', 0.9), 0.9)
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                               nesterov=True, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def fill_defaults(cfg):
    """填充配置默认值，仅保留Haar小波相关配置"""
    tr = cfg.setdefault('training', {})
    tr.setdefault('warmup_epochs', 5)
    tr.setdefault('amp', True)
    tr.setdefault('grad_clip', None)
    tr.setdefault('early_stop_patience', 20)
    tr.setdefault('momentum', 0.9)
    tr.setdefault('weight_decay', 1e-4)
    tr.setdefault('optimizer', 'adamw')
    tr.setdefault('scheduler', 'cosine')
    tr.setdefault('epochs', 100)
    tr.setdefault('batch_size', 32)
    tr.setdefault('num_workers', 4)
    tr.setdefault('gamma', 0.1)
    tr.setdefault('step_size', 30)

    lg = cfg.setdefault('logging', {})
    lg.setdefault('out_dir', 'outputs')
    lg.setdefault('tqdm', True)
    lg.setdefault('save_interval', 20)

    ev = cfg.setdefault('evaluation', {})
    ev.setdefault('topk', [1, 3])

    data = cfg.setdefault('data', {})
    data.setdefault('cache_in_memory', False)
    data.setdefault('file_ext', '.csv')
    data.setdefault('norm_mode', 'per_sample')
    data.setdefault('cmap', 'viridis')
    data.setdefault('resize', [64, 64])
    data.setdefault('max_time_len', 1100)
    data.setdefault('min_time_len', 200)
    data.setdefault('subcarriers', 52)
    data.setdefault('train_split', 0.75)
    data.setdefault('val_split', 0.10)
    data.setdefault('test_split', 0.15)
    # 仅保留Haar小波配置，移除小波类型选项
    data.setdefault('use_wavelet', False)  # 小波变换开关
    data.setdefault('wavelet_level', 4)  # 小波分解层数
    # zscore归一化配置
    data.setdefault('use_zscore', False)  # zscore开关
    # 添加STFT配置
    data.setdefault('use_stft', True)  # STFT开关

    cfg.setdefault('seed', 42)
    model = cfg.setdefault('model', {})
    model.setdefault('name', 'deeper_cnn')
    model.setdefault('in_channels', 3)
    model.setdefault('dropout', 0.0)


def get_scheduler(optimizer, cfg):
    tr = cfg['training']
    sch_name = str(tr.get('scheduler', 'none')).lower()
    epochs = tr['epochs']
    if sch_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sch_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=tr.get('step_size', 30),
            gamma=tr.get('gamma', 0.1)
        )
    else:
        return None


def apply_haar_wavelet_transform(data, level=2):
    """应用手动实现的Haar小波变换（不依赖pywt）"""
    # 确保数据是二维的
    if len(data.shape) != 2:
        raise ValueError(f"Haar小波变换需要二维数据，实际形状: {data.shape}")

    # 对每一行应用Haar小波分解
    transformed = []
    for row in data:
        coeffs = haar_wavelet_decompose(row, level=level)
        # 取近似系数和前两个细节系数
        cA = coeffs[0]
        cD1 = coeffs[1] if len(coeffs) > 1 else np.zeros_like(cA)
        cD2 = coeffs[2] if len(coeffs) > 2 else np.zeros_like(cA)

        # 调整长度并堆叠
        max_len = max(len(cA), len(cD1), len(cD2))
        cA_pad = np.pad(cA, (0, max_len - len(cA)), mode='edge')
        cD1_pad = np.pad(cD1, (0, max_len - len(cD1)), mode='edge')
        cD2_pad = np.pad(cD2, (0, max_len - len(cD2)), mode='edge')

        transformed.append(np.stack([cA_pad, cD1_pad, cD2_pad], axis=0))

    # 合并结果并归一化
    transformed = np.mean(transformed, axis=0)
    transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min() + 1e-8)
    return transformed


def apply_zscore_normalization(data, mean=None, std=None):
    """应用zscore归一化"""
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / (std + 1e-8), mean, std


def warmup_lr(optimizer, warmup_target_lr, step, warmup_steps):
    """线性预热学习率"""
    try:
        base_val = float(warmup_target_lr)
    except Exception:
        print(f"[WARN] warmup_lr: warmup_target_lr={warmup_target_lr} 解析失败，使用 1e-3")
        base_val = 1e-3
    denom = float(max(1, warmup_steps))
    lr = base_val * float(step) / denom
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_epoch(model, loader, device, criterion, optimizer, scaler, cfg, epoch, warmup_steps):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    topk_fields = cfg['evaluation']['topk']
    topk_acc_sum = [0.0] * len(topk_fields)
    pbar = tqdm(loader, disable=not cfg['logging']['tqdm'],
                desc=f"Train {epoch}")
    warmup_epochs = cfg['training']['warmup_epochs']
    warmup_target_lr = cfg['training']['warmup_target_lr']
    main_base_lr = cfg['training']['main_base_lr']
    global_step = epoch * len(loader)
    current_lr = optimizer.param_groups[0]['lr']
    for i, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device)
        step_idx = global_step + i + 1
        if epoch < warmup_epochs:
            current_lr = warmup_lr(optimizer, warmup_target_lr, step_idx, warmup_steps)
        else:
            if i == 0 and abs(optimizer.param_groups[0]['lr'] - main_base_lr) > 1e-12:
                for pg in optimizer.param_groups:
                    pg['lr'] = main_base_lr
                current_lr = main_base_lr
            else:
                current_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg['training']['amp']):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()

        if cfg['training']['grad_clip'] is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),
                                     cfg['training']['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        _, pred = outputs.max(1)
        total_correct += pred.eq(labels).sum().item()
        total_samples += batch_size
        topk_accs = accuracy_topk(outputs, labels, topk=topk_fields)
        for k_i, val in enumerate(topk_accs):
            topk_acc_sum[k_i] += val * batch_size / 100.0
        pbar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{100 * total_correct / total_samples:.2f}%",
            "lr": f"{current_lr:.2e}"
        })
    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    avg_topk = [100.0 * x / total_samples for x in topk_acc_sum]
    return avg_loss, avg_acc, avg_topk


@torch.no_grad()
def evaluate(model, loader, device, criterion, cfg, prefix="Eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    topk_fields = cfg['evaluation']['topk']
    topk_acc_sum = [0.0] * len(topk_fields)
    pbar = tqdm(loader, disable=not cfg['logging']['tqdm'],
                desc=prefix)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        _, pred = outputs.max(1)
        total_correct += pred.eq(labels).sum().item()
        total_samples += batch_size
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        topk_accs = accuracy_topk(outputs, labels, topk=topk_fields)
        for k_i, val in enumerate(topk_accs):
            topk_acc_sum[k_i] += val * batch_size / 100.0
    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    avg_topk = [100.0 * x / total_samples for x in topk_acc_sum]
    return avg_loss, avg_acc, avg_topk, all_preds, all_labels


def extract_user_from_filename(filename):
    """从文件名提取用户标识"""
    user_pattern = re.compile(r'user_(\d+)')
    match = user_pattern.search(filename)
    if match:
        user_id = match.group(1)
        return f"user_{user_id}"
    return None


def collect_action_user_files(all_files, classes):
    """按"动作-用户"二维分组收集文件"""
    action_user_files = {action: {} for action in classes}
    for file_path in all_files:
        action = os.path.basename(os.path.dirname(file_path))
        if action not in action_user_files:
            continue
        filename = os.path.basename(file_path)
        user = extract_user_from_filename(filename)
        if not user:
            print(f"[WARN] 无法从文件名 {filename} 提取用户，已忽略该文件")
            continue
        if user not in action_user_files[action]:
            action_user_files[action][user] = []
        action_user_files[action][user].append(file_path)
    print("\n[动作-用户分组统计]")
    total_users = set()
    for action, user_files in action_user_files.items():
        user_count = len(user_files)
        total_samples = sum(len(files) for files in user_files.values())
        print(f"  动作 {action}: {user_count} 个用户，共 {total_samples} 个样本")
        for user, files in user_files.items():
            print(f"    - {user}: {len(files)} 个样本")
        total_users.update(user_files.keys())
    print(f"[总计] 识别到 {len(total_users)} 个志愿者：{sorted(list(total_users))}")
    return action_user_files


def stratified_split(action_user_files, train_split, val_split, test_split, seed):
    """按"动作-用户"双层分层抽样划分数据集"""
    np.random.seed(seed)
    train_files = []
    val_files = []
    test_files = []
    print("\n[分层抽样详细结果]")
    for action, user_files in action_user_files.items():
        print(f"\n  动作 {action} 划分：")
        for user, files in user_files.items():
            shuffled_files = files.copy()
            np.random.shuffle(shuffled_files)
            total = len(shuffled_files)
            train_size = int(total * train_split)
            val_size = int(total * val_split)
            test_size = total - train_size - val_size

            if total >= 3:
                train_size = max(1, train_size)
                val_size = max(1, val_size)
                test_size = total - train_size - val_size
            elif total == 2:
                train_size = 1
                val_size = 1
                test_size = 0
            elif total == 1:
                train_size = 1
                val_size = 0
                test_size = 0

            user_train = shuffled_files[:train_size]
            user_val = shuffled_files[train_size:train_size + val_size]
            user_test = shuffled_files[train_size + val_size:]
            train_files.extend(user_train)
            val_files.extend(user_val)
            test_files.extend(user_test)
            print(f"    {user}: 训练={len(user_train)}, 验证={len(user_val)}, 测试={len(user_test)}")
    return train_files, val_files, test_files




def _to_float(name, val, default):
        if val is None:
            print(f"[WARN] {name} 未设置，采用默认 {default}")
            return default
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).strip())
        except Exception:
            print(f"[WARN] {name}={val} 不能转换为 float，采用默认 {default}")
            return default

def parse_lr(lr_cfg):
        """解析学习率配置，支持多种格式"""
        if isinstance(lr_cfg, (list, tuple)):
            if len(lr_cfg) == 0:
                print("[WARN] lr 列表为空，回退 1e-3")
                return 1e-3, 1e-3
            vals = []
            for v in lr_cfg:
                try:
                    vals.append(float(str(v).strip()))
                except Exception:
                    pass
            if len(vals) == 0:
                print("[WARN] lr 列表元素无法解析，回退 1e-3")
                return 1e-3, 1e-3
            if len(vals) == 1:
                return vals[0], vals[0]
            return vals[0], vals[-1]
        else:
            try:
                v = float(str(lr_cfg).strip())
                return v, v
            except Exception:
                print(f"[WARN] lr={lr_cfg} 解析失败，回退 1e-3")
                return 1e-3, 1e-3

def get_optimizer(parameters, cfg):
        tr = cfg['training']
        opt_name = str(tr.get('optimizer', 'adamw')).lower()
        lr = tr['main_base_lr']
        wd = _to_float('weight_decay', tr.get('weight_decay', 1e-4), 1e-4)
        if opt_name == 'adam':
            return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            momentum = _to_float('momentum', tr.get('momentum', 0.9), 0.9)
            return torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                                   nesterov=True, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

def fill_defaults(cfg):
        """填充配置默认值，仅保留Haar小波相关配置"""
        tr = cfg.setdefault('training', {})
        tr.setdefault('warmup_epochs', 5)
        tr.setdefault('amp', True)
        tr.setdefault('grad_clip', None)
        tr.setdefault('early_stop_patience', 20)
        tr.setdefault('momentum', 0.9)
        tr.setdefault('weight_decay', 1e-4)
        tr.setdefault('optimizer', 'adamw')
        tr.setdefault('scheduler', 'cosine')
        tr.setdefault('epochs', 100)
        tr.setdefault('batch_size', 32)
        tr.setdefault('num_workers', 4)
        tr.setdefault('gamma', 0.1)
        tr.setdefault('step_size', 30)

        lg = cfg.setdefault('logging', {})
        lg.setdefault('out_dir', 'outputs')
        lg.setdefault('tqdm', True)
        lg.setdefault('save_interval', 20)

        ev = cfg.setdefault('evaluation', {})
        ev.setdefault('topk', [1, 3])

        data = cfg.setdefault('data', {})
        data.setdefault('cache_in_memory', False)
        data.setdefault('file_ext', '.csv')
        data.setdefault('norm_mode', 'per_sample')
        data.setdefault('cmap', 'viridis')
        data.setdefault('resize', [64, 64])
        data.setdefault('max_time_len', 1100)
        data.setdefault('min_time_len', 200)
        data.setdefault('subcarriers', 52)
        data.setdefault('train_split', 0.75)
        data.setdefault('val_split', 0.10)
        data.setdefault('test_split', 0.15)
        # 仅保留Haar小波配置，移除小波类型选项
        data.setdefault('use_wavelet', False)  # 小波变换开关
        data.setdefault('wavelet_level', 4)  # 小波分解层数
        # zscore归一化配置
        data.setdefault('use_zscore', False)  # zscore开关
        # 添加STFT配置
        data.setdefault('use_stft', True)  # STFT开关
        data.setdefault('stft_nperseg', 128)  # 滑动窗口大小
        data.setdefault('stft_noverlap', 64)  # 窗口重叠大小
        data.setdefault('stft_nfft', 256)  # FFT点数

        cfg.setdefault('seed', 42)
        model = cfg.setdefault('model', {})
        model.setdefault('name', 'deeper_cnn')
        model.setdefault('in_channels', 3)
        model.setdefault('dropout', 0.0)

def get_scheduler(optimizer, cfg):
        tr = cfg['training']
        sch_name = str(tr.get('scheduler', 'none')).lower()
        epochs = tr['epochs']
        if sch_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif sch_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=tr.get('step_size', 30),
                gamma=tr.get('gamma', 0.1)
            )
        else:
            return None

def apply_haar_wavelet_transform(data, level=2):
        """应用手动实现的Haar小波变换（不依赖pywt）"""
        # 确保数据是二维的
        if len(data.shape) != 2:
            raise ValueError(f"Haar小波变换需要二维数据，实际形状: {data.shape}")

        # 对每一行应用Haar小波分解
        transformed = []
        for row in data:
            coeffs = haar_wavelet_decompose(row, level=level)
            # 取近似系数和前两个细节系数
            cA = coeffs[0]
            cD1 = coeffs[1] if len(coeffs) > 1 else np.zeros_like(cA)
            cD2 = coeffs[2] if len(coeffs) > 2 else np.zeros_like(cA)

            # 调整长度并堆叠
            max_len = max(len(cA), len(cD1), len(cD2))
            cA_pad = np.pad(cA, (0, max_len - len(cA)), mode='edge')
            cD1_pad = np.pad(cD1, (0, max_len - len(cD1)), mode='edge')
            cD2_pad = np.pad(cD2, (0, max_len - len(cD2)), mode='edge')

            transformed.append(np.stack([cA_pad, cD1_pad, cD2_pad], axis=0))

        # 合并结果并归一化
        transformed = np.mean(transformed, axis=0)
        transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min() + 1e-8)
        return transformed

def apply_zscore_normalization(data, mean=None, std=None):
        """应用zscore归一化"""
        if mean is None:
            mean = np.mean(data)
        if std is None:
            std = np.std(data)
        return (data - mean) / (std + 1e-8), mean, std

def warmup_lr(optimizer, warmup_target_lr, step, warmup_steps):
        """线性预热学习率"""
        try:
            base_val = float(warmup_target_lr)
        except Exception:
            print(f"[WARN] warmup_lr: warmup_target_lr={warmup_target_lr} 解析失败，使用 1e-3")
            base_val = 1e-3
        denom = float(max(1, warmup_steps))
        lr = base_val * float(step) / denom
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

def train_one_epoch(model, loader, device, criterion, optimizer, scaler, cfg, epoch, warmup_steps):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        topk_fields = cfg['evaluation']['topk']
        topk_acc_sum = [0.0] * len(topk_fields)
        pbar = tqdm(loader, disable=not cfg['logging']['tqdm'],
                    desc=f"Train {epoch}")
        warmup_epochs = cfg['training']['warmup_epochs']
        warmup_target_lr = cfg['training']['warmup_target_lr']
        main_base_lr = cfg['training']['main_base_lr']
        global_step = epoch * len(loader)
        current_lr = optimizer.param_groups[0]['lr']
        for i, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            step_idx = global_step + i + 1
            if epoch < warmup_epochs:
                current_lr = warmup_lr(optimizer, warmup_target_lr, step_idx, warmup_steps)
            else:
                if i == 0 and abs(optimizer.param_groups[0]['lr'] - main_base_lr) > 1e-12:
                    for pg in optimizer.param_groups:
                        pg['lr'] = main_base_lr
                    current_lr = main_base_lr
                else:
                    current_lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg['training']['amp']):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            if cfg['training']['grad_clip'] is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),
                                         cfg['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            _, pred = outputs.max(1)
            total_correct += pred.eq(labels).sum().item()
            total_samples += batch_size
            topk_accs = accuracy_topk(outputs, labels, topk=topk_fields)
            for k_i, val in enumerate(topk_accs):
                topk_acc_sum[k_i] += val * batch_size / 100.0
            pbar.set_postfix({
                "loss": f"{total_loss / total_samples:.4f}",
                "acc": f"{100 * total_correct / total_samples:.2f}%",
                "lr": f"{current_lr:.2e}"
            })
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        avg_topk = [100.0 * x / total_samples for x in topk_acc_sum]
        return avg_loss, avg_acc, avg_topk

@torch.no_grad()
def evaluate(model, loader, device, criterion, cfg, prefix="Eval"):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        topk_fields = cfg['evaluation']['topk']
        topk_acc_sum = [0.0] * len(topk_fields)
        pbar = tqdm(loader, disable=not cfg['logging']['tqdm'],
                    desc=prefix)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            _, pred = outputs.max(1)
            total_correct += pred.eq(labels).sum().item()
            total_samples += batch_size
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            topk_accs = accuracy_topk(outputs, labels, topk=topk_fields)
            for k_i, val in enumerate(topk_accs):
                topk_acc_sum[k_i] += val * batch_size / 100.0
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        avg_topk = [100.0 * x / total_samples for x in topk_acc_sum]
        return avg_loss, avg_acc, avg_topk, all_preds, all_labels

def extract_user_from_filename(filename):
        """从文件名提取用户标识"""
        user_pattern = re.compile(r'user_(\d+)')
        match = user_pattern.search(filename)
        if match:
            user_id = match.group(1)
            return f"user_{user_id}"
        return None

def collect_action_user_files(all_files, classes):
        """按"动作-用户"二维分组收集文件"""
        action_user_files = {action: {} for action in classes}
        for file_path in all_files:
            action = os.path.basename(os.path.dirname(file_path))
            if action not in action_user_files:
                continue
            filename = os.path.basename(file_path)
            user = extract_user_from_filename(filename)
            if not user:
                print(f"[WARN] 无法从文件名 {filename} 提取用户，已忽略该文件")
                continue
            if user not in action_user_files[action]:
                action_user_files[action][user] = []
            action_user_files[action][user].append(file_path)
        print("\n[动作-用户分组统计]")
        total_users = set()
        for action, user_files in action_user_files.items():
            user_count = len(user_files)
            total_samples = sum(len(files) for files in user_files.values())
            print(f"  动作 {action}: {user_count} 个用户，共 {total_samples} 个样本")
            for user, files in user_files.items():
                print(f"    - {user}: {len(files)} 个样本")
            total_users.update(user_files.keys())
        print(f"[总计] 识别到 {len(total_users)} 个志愿者：{sorted(list(total_users))}")
        return action_user_files

def stratified_split(action_user_files, train_split, val_split, test_split, seed):
        """按"动作-用户"双层分层抽样划分数据集"""
        np.random.seed(seed)
        train_files = []
        val_files = []
        test_files = []
        print("\n[分层抽样详细结果]")
        for action, user_files in action_user_files.items():
            print(f"\n  动作 {action} 划分：")
            for user, files in user_files.items():
                shuffled_files = files.copy()
                np.random.shuffle(shuffled_files)
                total = len(shuffled_files)
                train_size = int(total * train_split)
                val_size = int(total * val_split)
                test_size = total - train_size - val_size

                if total >= 3:
                    train_size = max(1, train_size)
                    val_size = max(1, val_size)
                    test_size = total - train_size - val_size
                elif total == 2:
                    train_size = 1
                    val_size = 1
                    test_size = 0
                elif total == 1:
                    train_size = 1
                    val_size = 0
                    test_size = 0

                user_train = shuffled_files[:train_size]
                user_val = shuffled_files[train_size:train_size + val_size]
                user_test = shuffled_files[train_size + val_size:]
                train_files.extend(user_train)
                val_files.extend(user_val)
                test_files.extend(user_test)
                print(f"    {user}: 训练={len(user_train)}, 验证={len(user_val)}, 测试={len(user_test)}")
        return train_files, val_files, test_files

    # 核心主函数（无嵌套，直接作为程序入口）
def main():
        # -------------------------- 核心参数配置 --------------------------
        config_path = "train_config.yaml"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resume_path = ""

        # -------------------------- 主逻辑 --------------------------
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在！路径：{config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        fill_defaults(cfg)

        # 打印配置文件和数据集文件的路径
        print(f"[INFO] 配置文件路径: {os.path.abspath(config_path)}")
        # 修改csi_dataset.py路径为Time-Frequency根目录
        csi_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "csi_dataset.py"))
        print(f"[INFO] csi_dataset.py路径: {csi_dataset_path}")
        print(f"[INFO] 配置文件是否存在: {os.path.exists(config_path)}")
        print(f"[INFO] csi_dataset.py是否存在: {os.path.exists(csi_dataset_path)}")

        # 解析学习率
        warmup_target_lr, main_base_lr = parse_lr(cfg['training'].get('lr', 1e-3))
        cfg['training']['warmup_target_lr'] = warmup_target_lr
        cfg['training']['main_base_lr'] = main_base_lr
        cfg['training']['lr_parsed'] = {
            'warmup_target_lr': warmup_target_lr,
            'main_base_lr': main_base_lr
        }
        print(f"[INFO] LR parse -> warmup_target={warmup_target_lr} main_base={main_base_lr}")

        # 解析分层抽样比例
        data_cfg = cfg['data']
        train_split = data_cfg['train_split']
        val_split = data_cfg['val_split']
        test_split = data_cfg['test_split']
        total_ratio = train_split + val_split + test_split
        if not np.isclose(total_ratio, 1.0, atol=1e-3):
            print(f"[WARN] 划分比例总和为 {total_ratio:.3f}，自动归一化到1.0")
            train_split /= total_ratio
            val_split /= total_ratio
            test_split /= total_ratio
        print(f"[INFO] 分层抽样比例 -> 训练={train_split:.3f}, 验证={val_split:.3f}, 测试={test_split:.3f}")

        # 打印数据处理配置（包含STFT）
        print(
            f"[INFO] 数据处理配置 -> Haar小波变换: {data_cfg['use_wavelet']}(level={data_cfg['wavelet_level']}), "
            f"zscore归一化: {data_cfg['use_zscore']}, STFT时频分析: {data_cfg.get('use_stft', True)}")

        # 固定随机种子
        set_seed(cfg['seed'])
        print(f"[INFO] 训练设备：{device}")

        # 创建输出目录
        out_dir = cfg['logging']['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "test_results"), exist_ok=True)
        print(f"[INFO] 输出文件夹完整路径: {os.path.abspath(out_dir)}")

        # 保存使用的配置
        with open(os.path.join(out_dir, 'used_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f)

        # 类别映射
        classes = cfg['data']['classes']
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # 1. 收集所有文件
        all_files = collect_files(cfg['data']['raw_root'], classes, file_ext=data_cfg['file_ext'])
        print(f"\n[INFO] 共发现样本文件数: {len(all_files)}")
        if len(all_files) == 0:
            print("[ERROR] 未找到任何样本文件，请检查raw_root路径和file_ext配置")
            return

        # 2. 按"动作-用户"二维分组
        action_user_files = collect_action_user_files(all_files, classes)

        # 3. 双层分层抽样划分数据集
        train_files, val_files, test_files = stratified_split(
            action_user_files,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=cfg['seed']
        )

        # 打印划分结果汇总
        print(f"\n[最终划分结果]")
        print(f"  训练集: {len(train_files)} 个样本")
        print(f"  验证集: {len(val_files)} 个样本")
        print(f"  测试集: {len(test_files)} 个样本")
        print(f"  总计: {len(train_files) + len(val_files) + len(test_files)} 个样本")

        # 4. 初始化数据转换器和预处理参数
        converter = CSIPseudoColorConverter(
            target_size=data_cfg['resize'],
            cmap=data_cfg['cmap'],
            norm_mode=data_cfg['norm_mode']
        )

        # 计算全局归一化参数
        global_mean = None
        global_std = None
        if data_cfg['norm_mode'] == 'global' or data_cfg['use_zscore']:
            print("\n[INFO] 计算全局归一化参数 ...")
            mats = []
            for i, fp in enumerate(train_files[:2000]):
                mats.append(load_csi_file(fp))

            if data_cfg['norm_mode'] == 'global':
                converter.fit_global(mats)

            if data_cfg['use_zscore']:
                all_data = np.concatenate([m.flatten() for m in mats])
                global_mean = np.mean(all_data)
                global_std = np.std(all_data)
                print(f"[INFO] zscore 全局参数 -> 均值: {global_mean:.4f}, 标准差: {global_std:.4f}")

        # 5. 创建数据集（使用STFT时频分析）
        train_ds = CSIDataset(
            train_files, class_to_idx, converter,
            min_time_len=data_cfg['min_time_len'],
            max_time_len=data_cfg['max_time_len'],
            subcarriers=data_cfg['subcarriers'],
            augment=True,
            cache=data_cfg['cache_in_memory'],
            wavelet_level=data_cfg['wavelet_level'],
            use_wavelet=data_cfg['use_wavelet'],
            use_zscore=data_cfg['use_zscore'],
            use_stft=data_cfg['use_stft'],
            stft_nperseg=data_cfg['stft_nperseg'],
            stft_noverlap=data_cfg['stft_noverlap'],
            stft_nfft=data_cfg['stft_nfft']
        )

        val_ds = CSIDataset(
            val_files, class_to_idx, converter,
            min_time_len=data_cfg['min_time_len'],
            max_time_len=data_cfg['max_time_len'],
            subcarriers=data_cfg['subcarriers'],
            augment=False,
            cache=data_cfg['cache_in_memory'],
            wavelet_level=data_cfg['wavelet_level'],
            use_wavelet=data_cfg['use_wavelet'],
            use_zscore=data_cfg['use_zscore'],
            use_stft=data_cfg['use_stft'],
            stft_nperseg=data_cfg['stft_nperseg'],
            stft_noverlap=data_cfg['stft_noverlap'],
            stft_nfft=data_cfg['stft_nfft']
        )

        test_ds = CSIDataset(
            test_files, class_to_idx, converter,
            min_time_len=data_cfg['min_time_len'],
            max_time_len=data_cfg['max_time_len'],
            subcarriers=data_cfg['subcarriers'],
            augment=False,
            cache=data_cfg['cache_in_memory'],
            wavelet_level=data_cfg['wavelet_level'],
            use_wavelet=data_cfg['use_wavelet'],
            use_zscore=data_cfg['use_zscore'],
            use_stft=data_cfg['use_stft'],
            stft_nperseg=data_cfg['stft_nperseg'],
            stft_noverlap=data_cfg['stft_noverlap'],
            stft_nfft=data_cfg['stft_nfft']
        )

        # 6. 创建数据加载器
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg['training']['batch_size'],
            shuffle=True,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True
        )

        # 7. 构建模型
        model = build_model(
            cfg['model']['name'],
            num_classes=len(classes),
            in_channels=cfg['model']['in_channels'],
            dropout=cfg['model']['dropout']
        ).to(device)
        print(f"\n[模型结构]")
        print(model)

        # 8. 初始化训练组件
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model.parameters(), cfg)
        scheduler = get_scheduler(optimizer, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['amp'])

        # 恢复训练状态
        start_epoch = 0
        best_val_acc = 0.0
        patience_counter = 0
        if resume_path and os.path.isfile(resume_path):
            print(f"\n[RESUME] 从 {resume_path} 恢复")
            ckpt = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optim'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt['epoch'] + 1
            best_val_acc = ckpt.get('best_val_acc', 0.0)

        # 初始化日志文件
        log_file = os.path.join(out_dir, "train_log.csv")
        if not os.path.exists(log_file):
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        warmup_steps = cfg['training']['warmup_epochs'] * len(train_loader)

        # 9. 训练循环
        print("\n[开始训练]")
        for epoch in range(start_epoch, cfg['training']['epochs']):
            tr_loss, tr_acc, tr_topk = train_one_epoch(
                model, train_loader, device, criterion, optimizer, scaler, cfg, epoch, warmup_steps
            )
            val_loss, val_acc, val_topk, _, _ = evaluate(
                model, val_loader, device, criterion, cfg, prefix="Validation"
            )

            if scheduler is not None and epoch >= cfg['training']['warmup_epochs']:
                scheduler.step()

            # 打印epoch日志
            print(f"[Epoch {epoch}] "
                  f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% "
                  f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}% "
                  f"| TopK(Val)={[round(x, 2) for x in val_topk]}")

            # 写入日志文件
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.4f},{val_loss:.6f},{val_acc:.4f}\n")

            # 早停与模型保存
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_val_acc': best_val_acc
                }, os.path.join(out_dir, "best_model.pth"))
                print(f"[SAVE] 新最佳模型 (ValAcc={val_acc:.2f}%)")
            else:
                patience_counter += 1

            if (epoch + 1) % cfg['logging']['save_interval'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_val_acc': best_val_acc
                }, os.path.join(out_dir, f"checkpoint_epoch_{epoch}.pth"))

            if patience_counter >= cfg['training']['early_stop_patience']:
                print("[EARLY STOP] 触发早停")
                break

        # 10. 测试评估
        print("\n[开始测试]")
        best_ckpt = torch.load(os.path.join(out_dir, "best_model.pth"), map_location=device)
        model.load_state_dict(best_ckpt['model'])
        test_loss, test_acc, test_topk, test_preds, test_labels = evaluate(
            model, test_loader, device, criterion, cfg, prefix="Test"
        )

        # 打印测试结果
        print(f"\n[测试结果汇总]")
        print(f"  Loss={test_loss:.4f} Acc={test_acc:.2f}% TopK={[round(x, 2) for x in test_topk]}")
        cm, report = compute_metrics(test_labels, test_preds, classes,
                                     out_dir=os.path.join(out_dir, "test_results"))
        print("\n[详细分类报告]")
        print(report)



# 程序入口（直接调用顶层main()）
if __name__ == "__main__":
    main()