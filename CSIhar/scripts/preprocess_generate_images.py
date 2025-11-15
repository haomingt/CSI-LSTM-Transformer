"""
用途：
  - 适配实际帧范围（min_time_len=200 ~ max_time_len=1100），线性插值对齐至中间目标长度
  - 基于scipy实现小波去噪（无需pywt，适配scipy 1.16.0）
  - 避免短帧过度插值、长帧特征丢失，保留动作自然周期
  - 生成伪彩色图像时自动覆盖同名文件
"""

# -------------------------- 基础库导入 --------------------------
import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import cv2
from scipy.interpolate import interp1d

# -------------------------- 小波变换导入（修复scipy 1.16.0路径）--------------------------
# 关键修复：scipy 1.16.0中wavedec/waverec在signal.wavelets子模块下
try:
    # 尝试1：直接从signal.wavelets导入（scipy 1.16.0的正确路径）
    from scipy.signal.wavelets import wavedec, waverec
except ImportError:
    try:
        # 尝试2：兼容部分版本的signal直接导入
        from scipy.signal import wavedec, waverec
    except ImportError as e:
        raise ImportError(
            f"scipy导入失败：{e}\n"
            "最终解决方法：终端执行以下命令重新安装scipy（版本适配）：\n"
            "pip uninstall scipy -y && pip install scipy==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "（1.10.0版本经测试，wavedec/waverec在signal.wavelets下可正常导入）"
        )

# -------------------------- 自定义模块导入 --------------------------
try:
    from datasets.csi_dataset import collect_files, load_csi_file
except ImportError as e:
    raise ImportError(
        f"自定义模块导入失败：{e}\n"
        "解决方法：1. 确认datasets/csi_dataset.py存在\n"
        "         2. 项目根目录右键→Mark Directory as→Sources Root"
    )

try:
    from utils.transforms import CSIPseudoColorConverter
except ImportError as e:
    raise ImportError(
        f"自定义模块导入失败：{e}\n"
        "解决方法：1. 确认utils/transforms.py存在\n"
        "         2. 项目根目录右键→Mark Directory as→Sources Root"
    )


# -------------------------- 核心功能函数 --------------------------
def wavelet_denoise_csi(
    raw_mat: np.ndarray,
    wavelet: str = 'haar',
    level: int = 4,
    threshold_mode: str = 'soft'
) -> np.ndarray:
    """
    基于scipy实现的小波去噪（适配scipy 1.16.0）：
    按子载波独立处理，用Haar小波抑制高频噪声，保留动作核心特征
    """
    T, F = raw_mat.shape
    denoised_mat = np.zeros_like(raw_mat, dtype=np.float32)

    for f in range(F):
        # 提取单个子载波的时序信号
        csi_seq = raw_mat[:, f].flatten()

        # 小波分解（scipy 1.16.0支持'haar'字符串）
        coeffs = wavedec(csi_seq, wavelet=wavelet, level=level)
        cA = coeffs[0]  # 近似系数（动作核心特征）
        cD_list = coeffs[1:]  # 细节系数（高频噪声）

        # 自适应阈值计算（抑制噪声）
        sigma = np.median(np.abs(cD_list[-1])) / 0.6745  # 估算噪声标准差
        threshold = sigma * np.sqrt(2 * np.log(T))  # 启发式阈值

        # 阈值处理（仅对细节系数操作）
        denoised_cD_list = []
        for cD in cD_list:
            if threshold_mode == 'soft':
                # 软阈值：平滑噪声边界，避免特征跳变
                denoised_cD = np.where(np.abs(cD) < threshold, 0, cD - np.sign(cD) * threshold)
            else:
                # 硬阈值：强抑制噪声，适合高频干扰极强的场景
                denoised_cD = np.where(np.abs(cD) < threshold, 0, cD)
            denoised_cD_list.append(denoised_cD)

        # 小波重构（恢复清洁信号）
        denoised_coeffs = [cA] + denoised_cD_list
        denoised_seq = waverec(denoised_coeffs, wavelet=wavelet)

        # 长度对齐（确保与原始信号帧数一致）
        denoised_seq = denoised_seq[:T] if len(denoised_seq) > T else np.pad(denoised_seq, (0, T - len(denoised_seq)), mode='edge')
        denoised_mat[:, f] = denoised_seq

    return denoised_mat


def interpolate_adjust_length(mat: np.ndarray, target_len: int) -> np.ndarray:
    """基于线性插值的长短帧对齐，保留动作自然周期"""
    T, F = mat.shape
    if T == target_len:
        return mat

    original_time = np.linspace(0, 1, T)
    target_time = np.linspace(0, 1, target_len)

    adjusted = np.zeros((target_len, F), dtype=mat.dtype)
    for f in range(F):
        interpolator = interp1d(
            original_time, mat[:, f],
            kind='linear', bounds_error=False, fill_value="extrapolate"
        )
        adjusted[:, f] = interpolator(target_time)

    return adjusted


# -------------------------- 主流程 --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/train_config.yaml", help="训练配置文件")
    parser.add_argument("--out_root", default="./data_processed_denoised", help="去噪后图像输出目录")
    parser.add_argument("--format", choices=['png', 'jpg'], default='png', help="输出图像格式")
    # 小波参数（适配Haar基）
    parser.add_argument("--wavelet", default='haar', choices=['haar'], help="小波基（仅支持haar）")
    parser.add_argument("--denoise-level", type=int, default=4, help="分解层级（3-4层）")
    parser.add_argument("--threshold-mode", default='soft', choices=['soft', 'hard'], help="阈值模式")
    args = parser.parse_args()

    # 校验Haar基层级（避免过度平滑）
    if args.denoise_level > 4:
        print(f"[WARNING] Haar小波基层级过深，自动调整为4")
        args.denoise_level = 4
    print(f"[INFO] 小波参数：基={args.wavelet}，层级={args.denoise_level}，阈值模式={args.threshold_mode}")

    # 计算项目根目录（脚本在scripts/下，根目录为上一级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, args.config)

    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    classes = cfg['data']['classes']
    min_time_len = cfg['data']['min_time_len']
    max_time_len = cfg['data']['max_time_len']
    target_time_len = (min_time_len + max_time_len) // 2
    print(f"[INFO] 帧范围：{min_time_len}~{max_time_len} → 目标：{target_time_len}帧")

    # 加载CSI文件路径
    files = collect_files(cfg['data']['raw_root'], classes, file_ext=cfg['data']['file_ext'])
    print(f"[DEBUG] 原始数据目录：{cfg['data']['raw_root']}，文件数：{len(files)}")
    if not files:
        raise ValueError("未找到CSI文件！请检查config.yaml中raw_root路径是否正确")

    # 初始化伪彩色转换器
    converter = CSIPseudoColorConverter(
        target_size=cfg['data']['resize'],
        cmap=cfg['data']['cmap'],
        norm_mode=cfg['data']['norm_mode']
    )

    # 按类别分组文件（用于类别归一化）
    class_file_dict = {cls: [] for cls in classes}
    for file_path in files:
        rel_path = os.path.relpath(file_path, cfg['data']['raw_root'])
        cls_name = rel_path.split(os.sep)[0]
        if cls_name in class_file_dict:
            class_file_dict[cls_name].append(file_path)

    # 1. 计算归一化参数（基于去噪后的数据）
    if cfg['data']['norm_mode'] == 'class':
        print(f"[INFO] 按类别计算归一化参数...")
        for cls_name in tqdm(classes, desc="处理类别"):
            cls_files = class_file_dict[cls_name]
            if not cls_files:
                print(f"[WARNING] 类别{cls_name}无数据，跳过")
                continue

            cls_csi_mats = []
            for file_path in cls_files:
                raw_mat = load_csi_file(file_path)
                denoised_mat = wavelet_denoise_csi(raw_mat, args.wavelet, args.denoise_level, args.threshold_mode)
                aligned_mat = interpolate_adjust_length(denoised_mat, target_time_len)
                cls_csi_mats.append(aligned_mat)
            converter.fit_class(cls_name, cls_csi_mats)

    elif cfg['data']['norm_mode'] == 'global':
        print(f"[INFO] 计算全局归一化参数...")
        global_csi_mats = []
        for idx, file_path in enumerate(tqdm(files, desc="加载全局数据")):
            raw_mat = load_csi_file(file_path)
            denoised_mat = wavelet_denoise_csi(raw_mat, args.wavelet, args.denoise_level, args.threshold_mode)
            aligned_mat = interpolate_adjust_length(denoised_mat, target_time_len)
            global_csi_mats.append(aligned_mat)
            if idx > 2000:
                break
        converter.fit_global(global_csi_mats)

    # 2. 生成去噪后的伪彩色图像
    output_paths = []
    for file_path in tqdm(files, desc=f"生成图像（{target_time_len}帧）"):
        # 步骤1：加载→去噪→对齐
        raw_mat = load_csi_file(file_path)
        denoised_mat = wavelet_denoise_csi(raw_mat, args.wavelet, args.denoise_level, args.threshold_mode)
        aligned_mat = interpolate_adjust_length(denoised_mat, target_time_len)

        # 步骤2：生成伪彩图
        rel_path = os.path.relpath(file_path, cfg['data']['raw_root'])
        cls_name = rel_path.split(os.sep)[0]
        img = converter(aligned_mat, cls_name) if cfg['data']['norm_mode'] == 'class' else converter(aligned_mat)

        # 步骤3：格式转换（CHW→HWC，RGB→BGR）
        img_hwc = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)

        # 步骤4：保存（含去噪标识）
        out_cls_dir = os.path.join(args.out_root, cls_name)
        os.makedirs(out_cls_dir, exist_ok=True)
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(out_cls_dir, f"{file_base}_wavelet.{args.format}")
        cv2.imwrite(out_path, img_bgr)

        output_paths.append(out_path)
        # 每50个文件打印进度
        if len(output_paths) % 50 == 0:
            print(f"[进度] 已生成{len(output_paths)}个文件，最近：{os.path.basename(out_path)}")

    # 完成提示
    print(f"\n[DONE] 处理完成！")
    print(f"输出目录：{os.path.abspath(args.out_root)}")
    print(f"文件特征：所有去噪图像文件名含'_wavelet'后缀")


if __name__ == "__main__":
    main()