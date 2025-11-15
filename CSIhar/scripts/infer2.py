import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# 导入项目中的必要模块
from utils.seed import set_seed
from datasets.csi_dataset import CSIDataset, collect_files, load_csi_file
from utils.transforms import CSIPseudoColorConverter
from models.csi_cnn import build_model
from utils.metrics import compute_metrics

# 默认文件路径 - 请根据实际情况修改这些路径
DEFAULT_CONFIG_PATH = "../configs/train_config.yaml"  # 默认配置文件路径
DEFAULT_MODEL_PATH = "./best_model/best_model.pth"  # 默认模型文件路径


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 填充默认配置（与train.py保持一致）
    fill_defaults(cfg)
    return cfg


def fill_defaults(cfg):
    """填充配置文件中的默认值"""
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
    data.setdefault('train_split', 0.75)
    data.setdefault('val_split', 0.10)
    data.setdefault('test_split', 0.15)

    data.setdefault('use_wavelet', True)
    data.setdefault('wavelet_level', 4)
    data.setdefault('wavelet_threshold_mode', 'soft')
    data.setdefault('use_zscore', True)

    cfg.setdefault('seed', 2024)
    model = cfg.setdefault('model', {})
    model.setdefault('name', 'simple_cnn')
    model.setdefault('in_channels', 3)
    model.setdefault('dropout', 0.0)


def extract_user_from_filename(filename):
    """
    从文件名提取用户标识（适配格式：user_XX_xxx，如user_1、user_5、user_10）
    返回：用户标识字符串（如"user_1"），提取失败返回None
    """
    user_pattern = re.compile(r'user_(\d+)')  # 匹配"user_"后接数字的模式
    match = user_pattern.search(filename)
    if match:
        user_id = match.group(1)
        return f"user_{user_id}"
    return None


def collect_action_user_files(all_files, classes):
    """
    按"动作-用户"二维分组收集文件，支持自动识别用户
    返回：{动作: {用户: [文件路径1, 文件路径2, ...]}}
    """
    action_user_files = {action: {} for action in classes}

    for file_path in all_files:
        # 1. 提取动作（基于文件所在文件夹）
        action = os.path.basename(os.path.dirname(file_path))  # 文件夹名即动作名
        if action not in action_user_files:
            continue  # 跳过非目标动作的文件

        # 2. 提取用户（基于文件名）
        filename = os.path.basename(file_path)
        user = extract_user_from_filename(filename)
        if not user:
            print(f"[WARN] 无法从文件名 {filename} 提取用户，已忽略该文件")
            continue

        # 3. 归入对应"动作-用户"组
        if user not in action_user_files[action]:
            action_user_files[action][user] = []
        action_user_files[action][user].append(file_path)

    # 打印分组统计信息
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
    """
    按"动作-用户"双层分层抽样划分数据集
    步骤：1. 对每个动作的每个用户单独划分；2. 合并所有动作的同类型子集
    返回：总训练集、总验证集、总测试集文件列表
    """
    np.random.seed(seed)  # 固定随机种子保证可复现
    train_files = []
    val_files = []
    test_files = []

    print("\n[分层抽样详细结果]")
    for action, user_files in action_user_files.items():
        print(f"\n  动作 {action} 划分：")

        # 每个用户的样本单独按比例划分
        for user, files in user_files.items():
            # 打乱样本顺序
            shuffled_files = files.copy()
            np.random.shuffle(shuffled_files)
            total = len(shuffled_files)

            # 计算划分边界（确保总和为total，处理取整误差）
            train_size = int(total * train_split)
            val_size = int(total * val_split)
            test_size = total - train_size - val_size

            # 避免极端情况（确保每个子集至少1个样本，若总样本≥3）
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

            # 划分当前用户的子集
            user_train = shuffled_files[:train_size]
            user_val = shuffled_files[train_size:train_size + val_size]
            user_test = shuffled_files[train_size + val_size:]

            # 合并到总集
            train_files.extend(user_train)
            val_files.extend(user_val)
            test_files.extend(user_test)

            # 打印当前用户的划分结果
            print(f"    {user}: 训练={len(user_train)}, 验证={len(user_val)}, 测试={len(user_test)}")

    return train_files, val_files, test_files


@torch.no_grad()
def evaluate_model(model, test_loader, device, classes):
    """评估模型在测试集上的性能"""
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        _, preds = outputs.max(1)

        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = 100.0 * total_correct / total_samples
    print(f"测试集准确率: {accuracy:.2f}%")

    # 计算混淆矩阵和分类报告
    cm, report = compute_metrics(all_labels, all_preds, classes)
    print("\n分类报告:")
    print(report)

    return accuracy, cm, all_labels, all_preds


def main():
    # 1. 使用默认配置文件路径
    config_path = DEFAULT_CONFIG_PATH
    print(f"使用默认配置文件: {config_path}")

    # 打印配置文件本地绝对路径
    config_abs_path = os.path.abspath(config_path)
    print(f"配置文件本地绝对路径: {config_abs_path}")

    if not config_path or not os.path.exists(config_path):
        print("未找到有效的配置文件，程序退出")
        return

    # 2. 加载配置
    cfg = load_config(config_path)
    print(f"已加载配置文件: {config_path}")

    # 3. 设置随机种子（复用配置中的seed）
    set_seed(cfg['seed'])
    print(f"使用随机种子: {cfg['seed']}")

    # 4. 使用默认模型文件路径
    model_path = DEFAULT_MODEL_PATH
    print(f"使用默认模型文件: {model_path}")

    # 打印模型文件本地绝对路径
    model_abs_path = os.path.abspath(model_path)
    print(f"模型文件本地绝对路径: {model_abs_path}")

    if not model_path or not os.path.exists(model_path):
        print("未找到有效的模型文件，程序退出")
        return
    print(f"已选择模型文件: {model_path}")

    # 5. 准备设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 6. 准备测试数据集
    data_cfg = cfg['data']
    classes = data_cfg['classes']
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # 解析分层抽样比例
    train_split = data_cfg['train_split']
    val_split = data_cfg['val_split']
    test_split = data_cfg['test_split']

    # 验证比例合法性
    total_ratio = train_split + val_split + test_split
    if not np.isclose(total_ratio, 1.0, atol=1e-3):
        print(f"[WARN] 划分比例总和为 {total_ratio:.3f}，自动归一化到1.0")
        train_split /= total_ratio
        val_split /= total_ratio
        test_split /= total_ratio
    print(f"[INFO] 分层抽样比例 -> 训练={train_split:.3f}, 验证={val_split:.3f}, 测试={test_split:.3f}")

    # 收集所有文件
    all_files = collect_files(data_cfg['raw_root'], classes, file_ext=data_cfg['file_ext'])
    print(f"共发现 {len(all_files)} 个样本文件")
    if len(all_files) == 0:
        print("[ERROR] 未找到任何样本文件，请检查raw_root路径和file_ext配置")
        return

    # 按"动作-用户"二维分组（自动识别用户）
    action_user_files = collect_action_user_files(all_files, classes)

    # 双层分层抽样划分数据集
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

    # 打印所有测试集文件名
    print("\n[测试集文件列表]")
    for i, file_path in enumerate(test_files, 1):
        print(f"{i}. {file_path}")

    # 创建数据转换器
    converter = CSIPseudoColorConverter(
        target_size=data_cfg['resize'],
        cmap=data_cfg['cmap'],
        norm_mode=data_cfg['norm_mode']
    )

    # 如果是全局归一化，需要计算全局参数
    if data_cfg['norm_mode'] == 'global':
        print("\n计算全局归一化参数...")
        mats = [load_csi_file(fp) for fp in train_files[:2000]]  # 使用训练集计算
        converter.fit_global(mats)

    # 创建测试数据集和数据加载器
    test_dataset = CSIDataset(
        test_files,
        class_to_idx,
        converter,
        max_time_len=data_cfg['max_time_len'],
        min_time_len=data_cfg['min_time_len'],
        subcarriers=data_cfg['subcarriers'],
        augment=False,
        cache=data_cfg['cache_in_memory'],
        use_wavelet=data_cfg['use_wavelet'],
        wavelet_level=data_cfg['wavelet_level'],
        wavelet_threshold_mode=data_cfg['wavelet_threshold_mode'],
        use_zscore=data_cfg['use_zscore']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )

    # 7. 加载模型
    model = build_model(
        cfg['model']['name'],
        num_classes=len(classes),
        in_channels=cfg['model']['in_channels'],
        dropout=cfg['model']['dropout']
    ).to(device)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"已加载模型权重，训练 epoch: {checkpoint.get('epoch', '未知')}")

    # 8. 评估模型
    print("\n开始评估模型...")
    accuracy, cm, labels, preds = evaluate_model(model, test_loader, device, classes)

    # 9. 保存并显示结果（可选）
    output_dir = cfg['logging']['out_dir']
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'final_test_report.txt'), 'w') as f:
        f.write(f"测试集准确率: {accuracy:.2f}%\n\n")
        f.write(classification_report(labels, preds, target_names=classes))

    print(f"评估报告已保存至: {os.path.join(output_dir, 'final_test_report.txt')}")


if __name__ == "__main__":
    main()