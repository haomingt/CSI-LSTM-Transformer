import os
import random
import numpy as np
import torch
import re
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def haar_wavelet_decompose(signal: np.ndarray, level: int) -> list:
    """Haar小波分解"""
    coeffs = []
    current = signal.copy()
    for _ in range(level):
        if len(current) % 2 != 0:
            current = np.append(current, current[-1])
        approx = (current[::2] + current[1::2]) / np.sqrt(2)
        detail = (current[::2] - current[1::2]) / np.sqrt(2)
        coeffs.append(detail)
        current = approx
    coeffs.append(current)
    return coeffs[::-1]


def haar_wavelet_reconstruct(coeffs: list) -> np.ndarray:
    """Haar小波重构"""
    current = coeffs[0].copy()
    for i in range(1, len(coeffs)):
        detail = coeffs[i]
        n = len(current)
        current = np.repeat(current, 2)[:len(detail) * 2]
        current[::2] = (current[::2] + detail) / np.sqrt(2)
        current[1::2] = (current[1::2] - detail) / np.sqrt(2)
    return current


def collect_files(root: str, classes: list, file_ext=".csv"):
    """收集文件路径并按类别-用户分组"""
    import glob
    res = {}
    for cls in classes:
        res[cls] = {}  # 格式: {类别: {用户: [文件列表]}}
        pattern = os.path.join(root, cls, f"*{file_ext}")
        for file_path in glob.glob(pattern):
            # 从文件名提取用户ID（假设格式为xxx_user_[ID]_xxx.csv）
            filename = os.path.basename(file_path)
            user_match = re.search(r'user_(\d+)', filename)
            if user_match:
                user_id = user_match.group(1)
                if user_id not in res[cls]:
                    res[cls][user_id] = []
                res[cls][user_id].append(file_path)
            else:
                print(f"警告: 无法从文件名 {filename} 提取用户ID，跳过该文件")
    return res


def split_dataset(grouped_files, train_ratio, val_ratio, test_ratio, seed=42):
    """按用户均衡划分数据集"""
    set_seed(seed)
    train_files = []
    val_files = []
    test_files = []

    for cls, users in grouped_files.items():
        for user, files in users.items():
            # 对每个用户的文件单独打乱
            shuffled = files.copy()
            random.shuffle(shuffled)
            n = len(shuffled)

            # 计算每个用户的样本分配数量
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            # 确保至少有一个样本分配到测试集
            n_test = max(1, n - n_train - n_val) if n > 0 else 0

            # 分配到不同集合
            train_files.extend(shuffled[:n_train])
            val_files.extend(shuffled[n_train:n_train + n_val])
            test_files.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    return train_files, val_files, test_files


def generate_classification_report(labels, preds, classes, save_path):
    """生成分类报告并保存为文件"""
    report = classification_report(
        labels, preds, target_names=classes,
        zero_division=0,
        output_dict=False
    )
    os.makedirs(save_path, exist_ok=True)
    report_path = os.path.join(save_path, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[INFO] 分类报告已保存至: {report_path}")
    return report


def plot_confusion_matrix(labels, preds, classes, save_path, title="Confusion Matrix (Normalized)"):
    """绘制混淆矩阵并保存"""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    img_path = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 混淆矩阵已保存至: {img_path}")
    return cm