"""
用途：生成基于傅里叶变换的频率-幅度特征图像
修改说明：使用傅里叶变换将时间帧转换为频率"帧"，输入给图像的是频率和幅度，保留52个子载波
"""

# -------------------------- 基础库导入 --------------------------
import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# -------------------------- 自定义模块导入 --------------------------
try:
    from datasets.csi_dataset import collect_files, load_csi_file
except ImportError as e:
    raise ImportError(
        f"自定义模块导入失败：{e}\n"
        "解决方法：1. 确认datasets/csi_dataset.py存在\n"
        "         2. 添加项目根目录到sys.path"
    )


# -------------------------- 核心配置参数 --------------------------
# 图像参数
TARGET_SIZE = (64, 64)
CMAP = 'viridis'  # 建议尝试 'plasma' 以获得更鲜明的对比

# 数据参数
NUM_SUB_CARRIERS = 52  # 固定52个子载波
# 傅里叶变换后保留的频率点数量(取一半，因为FFT结果对称)
NUM_FREQ_BINS = 128


# -------------------------- 核心功能函数 --------------------------
def time_to_freq_domain(mat: np.ndarray) -> np.ndarray:
    """
    将时域信号转换为频域信号
    输入: (时间帧数量, 52)的CSI矩阵
    输出: (频率点数量, 52)的频域矩阵，值为幅度
    """
    T, F = mat.shape

    # 确保输入是52个子载波
    if F != NUM_SUB_CARRIERS:
        raise ValueError(f"子载波数量应为{NUM_SUB_CARRIERS}，实际为{F}")

    # 初始化频域矩阵
    freq_mat = np.zeros((NUM_FREQ_BINS, NUM_SUB_CARRIERS), dtype=np.float32)

    # 对每个子载波进行傅里叶变换
    for subcarrier in range(NUM_SUB_CARRIERS):
        # 提取单个子载波的时域信号
        time_signal = mat[:, subcarrier]

        # 应用傅里叶变换
        fft_result = np.fft.fft(time_signal)

        # 取幅度并进行对数转换以增强低幅度特征
        magnitude = np.abs(fft_result)
        magnitude = np.log1p(magnitude)  # log(1 + x) 避免零值问题

        # 只保留前半部分(FFT结果对称)并调整到目标频率点数
        if len(magnitude) // 2 < NUM_FREQ_BINS:
            # 如果频率点不足，进行插值
            freq_points = np.linspace(0, len(magnitude)//2 - 1, NUM_FREQ_BINS)
            magnitude = np.interp(freq_points, np.arange(len(magnitude)//2), magnitude[:len(magnitude)//2])
        else:
            # 如果频率点过多，取前NUM_FREQ_BINS个
            magnitude = magnitude[:NUM_FREQ_BINS]

        # 存储到频域矩阵
        freq_mat[:, subcarrier] = magnitude

    return freq_mat


def convert_freq_amp_to_image(freq_amp_data):
    """
    将频率-幅度数据转换为图像
    输入: (频率点数量, 52)的频域幅度矩阵
    输出: 转换后的图像
    """
    # 检查输入维度
    if freq_amp_data.ndim != 2:
        raise ValueError(f"频域幅度数据维度异常（需2维：频率×子载波）：{freq_amp_data.ndim}")

    if freq_amp_data.shape[1] != NUM_SUB_CARRIERS:
        raise ValueError(f"子载波数量异常（需{NUM_SUB_CARRIERS}列）：{freq_amp_data.shape[1]}")

    # 归一化到[0, 1]范围
    min_val = np.min(freq_amp_data)
    max_val = np.max(freq_amp_data)
    if max_val - min_val < 1e-8:
        # 处理常量信号的特殊情况
        normalized_data = np.zeros_like(freq_amp_data)
    else:
        normalized_data = (freq_amp_data - min_val) / (max_val - min_val)

    # 应用颜色映射 - 将频率-幅度数据转换为伪彩色图像
    cmap = plt.get_cmap(CMAP)
    colored_image = (cmap(normalized_data)[:, :, :3] * 255).astype(np.uint8)  # 仅保留RGB通道

    # 调整到目标尺寸
    return cv2.resize(colored_image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def force_collect_files(raw_root):
    """强制遍历所有子目录，自动检测常见扩展名的数据文件"""
    file_list = []
    detected_ext = None

    for root, _, files in os.walk(raw_root):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ['.csv', '.npy', '.txt', '.dat', '.mat']:
                full_path = os.path.join(root, file)
                file_list.append(full_path)
                detected_ext = file_ext  # 记录实际检测到的扩展名

    return file_list, detected_ext


# -------------------------- 主流程 --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/train_config.yaml", help="训练配置文件（可选）")
    parser.add_argument("--format", choices=['png', 'jpg'], default='png', help="输出图像格式")
    parser.add_argument("--raw_root", default="../data_raw_3", help="原始数据目录")
    args = parser.parse_args()

    # -------------------------- 1. 输出路径初始化 --------------------------
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    out_root = os.path.join(script_dir, "data_processed_freq_amp")  # 频率-幅度图像输出目录
    os.makedirs(out_root, exist_ok=True)
    print(f"[INFO] 输出目录：{os.path.abspath(out_root)}")

    # -------------------------- 2. 关键参数打印 --------------------------
    print(f"[INFO] 关键参数：")
    print(f"[INFO] - 目标尺寸：{TARGET_SIZE}")
    print(f"[INFO] - 子载波数量：{NUM_SUB_CARRIERS}")
    print(f"[INFO] - 频率点数量：{NUM_FREQ_BINS}")
    print(f"[INFO] - 数据目录：{args.raw_root}")

    # -------------------------- 3. 搜索数据文件 --------------------------
    raw_root_abs = os.path.abspath(args.raw_root)
    print(f"[INFO] 数据目录（绝对路径）：{raw_root_abs}")

    if not os.path.exists(raw_root_abs):
        raise FileNotFoundError(f"数据目录不存在：{raw_root_abs}")

    print(f"[INFO] 强制搜索所有子目录，自动检测常见扩展名...")
    files, detected_ext = force_collect_files(raw_root_abs)

    if not files:
        print(f"\n[DEBUG] 数据目录结构（仅显示一级子目录）：")
        for item in os.listdir(raw_root_abs):
            item_path = os.path.join(raw_root_abs, item)
            if os.path.isdir(item_path):
                sub_files = os.listdir(item_path)
                print(f"  [目录] {item} → 包含{len(sub_files)}个文件")
                for i, sub_file in enumerate(sub_files[:3]):
                    print(f"    - {sub_file}")
                if len(sub_files) > 3:
                    print(f"    - ...（共{len(sub_files)}个文件）")

        raise ValueError(
            f"未找到任何数据文件！问题排查：\n"
            f"1. 请确认数据文件在指定目录的子目录中\n"
            f"2. 检查文件扩展名是否被支持"
        )

    print(f"[INFO] 搜索成功！")
    print(f"[INFO] - 找到{len(files)}个文件")
    print(f"[INFO] - 检测到的文件扩展名：{detected_ext}")
    print(f"[INFO] - 示例文件：{os.path.basename(files[0])}")

    # -------------------------- 4. 生成频率-幅度图像 --------------------------
    output_paths = []
    for file_path in tqdm(files, desc="生成频率-幅度图像"):
        try:
            # 加载CSI数据
            raw_mat = load_csi_file(file_path)
            if raw_mat is None or raw_mat.size == 0:
                print(f"[WARNING] 空文件：{os.path.basename(file_path)}")
                continue
            if raw_mat.ndim != 2:
                print(f"[WARNING] 数据维度异常（需2维：时间×子载波）：{os.path.basename(file_path)}")
                continue

            # 转换为频域并生成图像
            freq_amp_mat = time_to_freq_domain(raw_mat)
            image = convert_freq_amp_to_image(freq_amp_mat)

            # 按类别保存
            rel_path = os.path.relpath(file_path, raw_root_abs)
            cls_name = rel_path.split(os.sep)[0]
            out_cls_dir = os.path.join(out_root, cls_name)
            os.makedirs(out_cls_dir, exist_ok=True)

            # 生成输出文件名（加freq_amp后缀区分）
            file_base = os.path.splitext(os.path.basename(file_path))[0]
            out_path = os.path.join(out_cls_dir, f"{file_base}_freq_amp.{args.format}")

            # 保存图像（转换为BGR格式，适应OpenCV）
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if cv2.imwrite(out_path, image_bgr):
                output_paths.append(out_path)
            else:
                print(f"[WARNING] 保存失败（权限不足）：{os.path.basename(out_path)}")

        except Exception as e:
            print(f"[ERROR] 处理{os.path.basename(file_path)}时出错：{str(e)}")
            continue

    # -------------------------- 5. 完成提示 --------------------------
    print(f"\n[DONE] 处理完成！")
    print(f"✅ 输出目录：{os.path.abspath(out_root)}")
    print(f"✅ 生成文件数：{len(output_paths)}/{len(files)}")
    if output_paths:
        print(f"✅ 类别分布：{set([os.path.basename(os.path.dirname(p)) for p in output_paths])}")


if __name__ == "__main__":
    main()
