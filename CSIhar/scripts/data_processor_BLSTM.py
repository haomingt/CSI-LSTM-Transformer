import numpy as np
import glob
import os
import csv
from sklearn.model_selection import train_test_split
import argparse


# ==========================
# 1. 提取时序序列（仅0-255归一化，不做滤波）
# ==========================
def merge_csi_sequence(csifile, seq_len=60, step=30):
    """
    从CSI文件提取序列：0-255归一化→滑动窗口截取（移除低通滤波步骤）
    - seq_len: 序列长度60（保持与之前一致）
    - step: 滑动窗口步长30（避免样本冗余）
    """
    csi = []
    with open(csifile, 'r', encoding='utf-8') as csif:
        reader = csv.reader(csif)
        for line in reader:
            if not line:
                continue
            # 提取前52列CSI幅度特征（原始信号，不做滤波）
            line_array = np.array([float(v.strip()) for v in line[:52]])
            csi.append(line_array)

    if not csi:
        return np.array([])

    # 转为numpy数组（shape: (时间步, 52)）
    csi = np.array(csi)

    # 仅保留归一化步骤（移除低通滤波）
    csi_min = np.min(csi)
    csi_max = np.max(csi)
    # 避免除以零（极端情况）
    if csi_max - csi_min < 1e-9:
        csi_normalized = np.full_like(csi, 128.0)
    else:
        csi_normalized = 255 * (csi - csi_min) / (csi_max - csi_min)
    # 转为uint8（1字节/元素，控制显存占用）
    csi_normalized = csi_normalized.astype(np.uint8)

    # 滑动窗口截取时序序列（保持不变）
    sequences = []
    index = 0
    while index + seq_len <= csi_normalized.shape[0]:
        window_csi = csi_normalized[index:index + seq_len, :]
        sequences.append(window_csi[np.newaxis, ...])
        index += step

    return np.concatenate(sequences, axis=0) if sequences else np.array([])


# ==========================
# 2. 按标签提取CSI序列
# ==========================
def extract_csi_by_label_blstm(raw_folder, target_label, all_labels, seq_len=60, step=30, save=True):
    print(f"=== 提取活动 '{target_label}' 的CSI序列（无滤波，仅归一化）===")
    target_label_lower = target_label.lower()
    all_labels_lower = [l.lower() for l in all_labels]

    if target_label_lower not in all_labels_lower:
        raise ValueError(f"非法标签 '{target_label}'，合法标签：{all_labels}")
    target_label_idx = all_labels_lower.index(target_label_lower)

    # 匹配该活动的所有CSV文件
    csi_file_pattern = os.path.join(raw_folder, target_label, "user_*.csv")
    csi_files = sorted(glob.glob(csi_file_pattern))
    if not csi_files:
        raise FileNotFoundError(f"未找到 {target_label} 的数据文件，路径：{csi_file_pattern}")

    all_sequences = []
    for file_idx, csi_file in enumerate(csi_files, 1):
        print(f"处理文件 {file_idx}/{len(csi_files)}: {os.path.basename(csi_file)}")
        # 提取序列（不做滤波）
        file_sequences = merge_csi_sequence(
            csi_file, seq_len=seq_len, step=step
        )
        if file_sequences.size == 0:
            print(f"警告：{os.path.basename(csi_file)} 未提取到序列，跳过")
            continue
        all_sequences.append(file_sequences)
        print(f"  提取到 {file_sequences.shape[0]} 个序列，当前活动总计 {sum(s.shape[0] for s in all_sequences)} 个")

    if not all_sequences:
        raise RuntimeError(f"{target_label} 未提取到有效序列")

    # 合并序列并生成标签
    seq_arr = np.concatenate(all_sequences, axis=0)
    label_arr = np.full((seq_arr.shape[0],), target_label_idx, dtype=np.int32)

    # 保存单活动数据
    if save:
        save_dir = "processed_data_BLSTM_no_filter"  # 新目录，避免与滤波数据混淆
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"csi_blstm_{target_label_lower}_seq.npz")
        np.savez_compressed(save_path, sequences=seq_arr, labels=label_arr)
        print(f"✅ {target_label} 数据保存至：{os.path.abspath(save_path)}")

    print(f"=== {target_label} 提取完成，共 {seq_arr.shape[0]} 个序列（shape: {seq_arr.shape}）===\n")
    return seq_arr, label_arr


# ==========================
# 3. 生成训练/测试集
# ==========================
def prepare_data_blstm(raw_folder, all_labels, seq_len=60, step=30, train_ratio=0.75, random_seed=379, save=True):
    print("=" * 60)
    print("开始CSI数据预处理（无低通滤波，仅归一化）")
    print(f"参数：序列长度={seq_len}，滑动步长={step}")
    print("=" * 60 + "\n")

    seq_list, label_list = [], []
    for target_label in all_labels:
        seq, label = extract_csi_by_label_blstm(
            raw_folder=raw_folder,
            target_label=target_label,
            all_labels=all_labels,
            seq_len=seq_len,
            step=step,
            save=save
        )
        seq_list.append(seq)
        label_list.append(label)

    # 合并所有活动数据
    all_seq = np.concatenate(seq_list, axis=0)
    all_label = np.concatenate(label_list, axis=0)

    # 分层划分训练/测试集（保持标签分布）
    x_train, x_test, y_train, y_test = train_test_split(
        all_seq, all_label,
        test_size=1 - train_ratio,
        random_state=random_seed,
        stratify=all_label
    )

    # 打乱训练集顺序
    train_idx = np.random.permutation(len(x_train))
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    # 打印结果
    print("=== 数据集划分完成 ===")
    print(f"总样本数：{all_seq.shape[0]}（训练集{len(x_train)} + 测试集{len(x_test)}）")
    print(f"单样本形状：{x_train.shape[1:]}（时间步×子载波）")
    print(f"数据类型：{x_train.dtype}（uint8，0-255）")
    print(f"各活动样本数：")
    for idx, label in enumerate(all_labels):
        train_count = np.sum(y_train == idx)
        test_count = np.sum(y_test == idx)
        print(f"  {label}：训练集{train_count} + 测试集{test_count} = 总计{train_count + test_count}")

    # 保存最终数据集（新路径，避免覆盖）
    if save:
        save_dir = "processed_data_BLSTM_no_filter"
        save_path = os.path.join(save_dir, "csi_blstm_train_test.npz")
        np.savez_compressed(
            save_path,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test
        )
        print(f"\n✅ 训练/测试集保存至：{os.path.abspath(save_path)}")

    return x_train, y_train, x_test, y_test


# ==========================
# 4. 主函数（命令行调用）
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSI信号预处理（无低通滤波，仅归一化，适配BLSTM）')
    parser.add_argument('--raw_folder', type=str,
                        default='/home/chenjun/python_codes/WiFi_sensing/data_raw',
                        help='原始CSI数据根目录（每个活动一个子文件夹）')
    parser.add_argument('--seq_len', type=int, default=60,
                        help='时序序列长度（保持60）')
    parser.add_argument('--step', type=int, default=30,
                        help='滑动窗口步长（保持30）')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                        help='训练集占比')
    parser.add_argument('--random_seed', type=int, default=379,
                        help='随机种子（保证可复现）')
    args = parser.parse_args()

    # 活动标签（与原始数据文件夹名对应）
    all_labels = ["lie_down", "fall", "bend", "run", "sitdown", "standup", "walk"]

    # 验证原始数据目录
    if not os.path.exists(args.raw_folder):
        raise FileNotFoundError(f"原始数据目录不存在：{args.raw_folder}")

    # 执行预处理
    x_train, y_train, x_test, y_test = prepare_data_blstm(
        raw_folder=args.raw_folder,
        all_labels=all_labels,
        seq_len=args.seq_len,
        step=args.step,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        save=True
    )
