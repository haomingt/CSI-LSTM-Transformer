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
from datasets.csi_dataset import collect_files, CSIDataset, load_csi_file
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
    data.setdefault('max_time_len', 512)
    data.setdefault('min_time_len', 0)
    data.setdefault('subcarriers', 30)
    data.setdefault('train_split', 0.7)
    data.setdefault('val_split', 0.15)
    data.setdefault('test_split', 0.15)
    data.pop('z_threshold', None)
    data.pop('filter_window', None)

    cfg.setdefault('seed', 42)
    model = cfg.setdefault('model', {})
    model.setdefault('name', 'simple_cnn')
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


def warmup_lr(optimizer, warmup_target_lr, step, warmup_steps):
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
    user_pattern = re.compile(r'user_(\d+)')
    match = user_pattern.search(filename)
    if match:
        user_id = match.group(1)
        return f"user_{user_id}"
    return None


def collect_action_user_files(all_files, classes):
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


def split_action_user_data(action_user_files, train_split, val_split, test_split, seed):
    np.random.seed(seed)
    # 按用户组存储测试集
    group_test_files = {
        'paper_group': [],
        'self_test_group': []
    }

    # 全局训练集和验证集
    all_train = []
    all_val = []

    # 定义用户分组
    user_groups = {
        'paper_group': ['user_1', 'user_2', 'user_3'],
        'self_test_group': ['user_5']
    }

    print("\n[单动作单志愿者数据划分]")
    for action, user_files in action_user_files.items():
        print(f"\n  动作 {action} 划分：")
        for user, files in user_files.items():
            shuffled = files.copy()
            np.random.shuffle(shuffled)
            total = len(shuffled)

            # 计算划分大小
            train_size = int(total * train_split)
            val_size = int(total * val_split)
            test_size = total - train_size - val_size

            # 确保每个子集合理分配
            if total >= 3:
                train_size = max(1, train_size)
                val_size = max(1, val_size)
                test_size = total - train_size - val_size
            elif total == 2:
                train_size, val_size, test_size = 1, 1, 0
            elif total == 1:
                train_size, val_size, test_size = 1, 0, 0

            # 划分数据集
            train = shuffled[:train_size]
            val = shuffled[train_size:train_size + val_size]
            test = shuffled[train_size + val_size:]

            # 合并到全局训练集和验证集
            all_train.extend(train)
            all_val.extend(val)

            # 根据用户所属组添加到相应的测试集
            if user in user_groups['paper_group']:
                group_test_files['paper_group'].extend(test)
            elif user in user_groups['self_test_group']:
                group_test_files['self_test_group'].extend(test)

            print(f"    {user}: 训练={len(train)}, 验证={len(val)}, 测试={len(test)}")

    return all_train, all_val, group_test_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/train_config.yaml",
                        help="训练配置文件路径")
    parser.add_argument("--resume", type=str, default="",
                        help="可选：载入已保存 checkpoint 路径继续训练")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    fill_defaults(cfg)

    # 解析学习率
    warmup_target_lr, main_base_lr = parse_lr(cfg['training'].get('lr', 1e-3))
    cfg['training']['warmup_target_lr'] = warmup_target_lr
    cfg['training']['main_base_lr'] = main_base_lr
    cfg['training']['lr_parsed'] = {
        'warmup_target_lr': warmup_target_lr,
        'main_base_lr': main_base_lr
    }
    print(f"[INFO] LR parse -> warmup_target={warmup_target_lr} main_base={main_base_lr}")

    # 解析数据划分比例
    train_split = cfg['data']['train_split']
    val_split = cfg['data']['val_split']
    test_split = cfg['data']['test_split']

    total_ratio = train_split + val_split + test_split
    if not np.isclose(total_ratio, 1.0, atol=1e-3):
        print(f"[WARN] 划分比例总和为 {total_ratio:.3f}，自动归一化到1.0")
        train_split /= total_ratio
        val_split /= total_ratio
        test_split /= total_ratio
    print(f"[INFO] 数据划分比例 -> 训练={train_split:.3f}, 验证={val_split:.3f}, 测试={test_split:.3f}")

    set_seed(cfg['seed'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    out_dir = cfg['logging']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test_results_paper_group"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test_results_self_test_group"), exist_ok=True)

    # 保存使用的配置
    with open(os.path.join(out_dir, 'used_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)

    # 类别映射
    classes = cfg['data']['classes']
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # 收集所有文件
    all_files = collect_files(cfg['data']['raw_root'], classes, file_ext=cfg['data']['file_ext'])
    print(f"[INFO] 共发现样本文件数: {len(all_files)}")
    if len(all_files) == 0:
        print("[ERROR] 未找到任何样本文件，请检查raw_root路径和file_ext配置")
        return

    # 按动作-用户分组
    action_user_files = collect_action_user_files(all_files, classes)

    # 按单动作单志愿者划分数据，然后合并训练集、验证集，按用户组保留测试集
    train_files, val_files, group_test_files = split_action_user_data(
        action_user_files,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=cfg['seed']
    )

    # 打印最终划分结果
    print(f"\n[最终划分结果]")
    print(f"  训练集: {len(train_files)} 个样本")
    print(f"  验证集: {len(val_files)} 个样本")
    print(f"  论文组测试集: {len(group_test_files['paper_group'])} 个样本")
    print(f"  自测组测试集: {len(group_test_files['self_test_group'])} 个样本")
    print(
        f"  总计: {len(train_files) + len(val_files) + len(group_test_files['paper_group']) + len(group_test_files['self_test_group'])} 个样本")

    # 初始化转换器
    converter = CSIPseudoColorConverter(
        target_size=cfg['data']['resize'],
        cmap=cfg['data']['cmap'],
        norm_mode=cfg['data']['norm_mode']
    )

    # 计算全局归一化参数
    if cfg['data']['norm_mode'] == 'global':
        print("[INFO] 计算全局 min/max 用于 global 归一化 ...")
        mats = []
        all_train_val = train_files + val_files
        for i, fp in enumerate(all_train_val[:2000]):  # 限制样本量
            mats.append(load_csi_file(fp))
        converter.fit_global(mats)

    # 创建数据集
    train_ds = CSIDataset(train_files, class_to_idx, converter,
                          max_time_len=cfg['data']['max_time_len'],
                          subcarriers=cfg['data']['subcarriers'],
                          augment=True,
                          cache=cfg['data']['cache_in_memory'],
                          min_time_len=cfg['data']['min_time_len'])

    val_ds = CSIDataset(val_files, class_to_idx, converter,
                        max_time_len=cfg['data']['max_time_len'],
                        subcarriers=cfg['data']['subcarriers'],
                        augment=False,
                        cache=cfg['data']['cache_in_memory'],
                        min_time_len=cfg['data']['min_time_len'])

    paper_test_ds = CSIDataset(group_test_files['paper_group'], class_to_idx, converter,
                               max_time_len=cfg['data']['max_time_len'],
                               subcarriers=cfg['data']['subcarriers'],
                               augment=False,
                               cache=cfg['data']['cache_in_memory'],
                               min_time_len=cfg['data']['min_time_len'])

    self_test_ds = CSIDataset(group_test_files['self_test_group'], class_to_idx, converter,
                              max_time_len=cfg['data']['max_time_len'],
                              subcarriers=cfg['data']['subcarriers'],
                              augment=False,
                              cache=cfg['data']['cache_in_memory'],
                              min_time_len=cfg['data']['min_time_len'])

    # 创建数据加载器
    train_loader = DataLoader(train_ds,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['training']['num_workers'],
                              pin_memory=True)

    val_loader = DataLoader(val_ds,
                            batch_size=cfg['training']['batch_size'],
                            shuffle=False,
                            num_workers=cfg['training']['num_workers'],
                            pin_memory=True)

    paper_test_loader = DataLoader(paper_test_ds,
                                   batch_size=cfg['training']['batch_size'],
                                   shuffle=False,
                                   num_workers=cfg['training']['num_workers'],
                                   pin_memory=True)

    self_test_loader = DataLoader(self_test_ds,
                                  batch_size=cfg['training']['batch_size'],
                                  shuffle=False,
                                  num_workers=cfg['training']['num_workers'],
                                  pin_memory=True)

    # 构建模型
    model = build_model(cfg['model']['name'],
                        num_classes=len(classes),
                        in_channels=cfg['model']['in_channels'],
                        dropout=cfg['model']['dropout']).to(device)
    print(f"\n[模型结构]")
    print(model)

    # 初始化训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['amp'])

    # 恢复训练状态
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"[RESUME] 从 {args.resume} 恢复")
        ckpt = torch.load(args.resume, map_location='cpu')
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

    # 训练循环
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

        print(f"[Epoch {epoch}] "
              f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}% "
              f"| TopK(Val)={[round(x, 2) for x in val_topk]}")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.4f},{val_loss:.6f},{val_acc:.4f}\n")

        # 早停和模型保存
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

    # 测试评估
    print("\n[开始测试]")
    best_ckpt = torch.load(os.path.join(out_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt['model'])

    # 论文组测试集评估
    print("\n[论文组测试集评估 - user_1, user_2, user_3]")
    test_loss, test_acc, test_topk, test_preds, test_labels = evaluate(
        model, paper_test_loader, device, criterion, cfg, prefix="Paper Group Test"
    )
    print(f"[论文组测试结果] Loss={test_loss:.4f} Acc={test_acc:.2f}% TopK={[round(x, 2) for x in test_topk]}")
    cm, report = compute_metrics(test_labels, test_preds, classes,
                                 out_dir=os.path.join(out_dir, "test_results_paper_group"))
    print(report)

    # 自测组测试集评估
    print("\n[自测组测试集评估 - user_5]")
    test_loss, test_acc, test_topk, test_preds, test_labels = evaluate(
        model, self_test_loader, device, criterion, cfg, prefix="Self Test Group Test"
    )
    print(f"[自测组测试结果] Loss={test_loss:.4f} Acc={test_acc:.2f}% TopK={[round(x, 2) for x in test_topk]}")
    cm, report = compute_metrics(test_labels, test_preds, classes,
                                 out_dir=os.path.join(out_dir, "test_results_self_test_group"))
    print(report)


if __name__ == "__main__":
    main()