import os
import glob
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from utils.transforms import CSIPseudoColorConverter, random_augment


# -------------------------- 新增：小波去噪相关函数 --------------------------
def haar_wavelet_decompose(signal: np.ndarray, level: int) -> list:
    """Haar小波分解（纯numpy实现）"""
    coeffs = []
    current = signal.copy()
    for _ in range(level):
        # 确保长度为偶数
        if len(current) % 2 != 0:
            current = np.append(current, current[-1])
        # 分解为近似系数和细节系数
        approx = (current[::2] + current[1::2]) / np.sqrt(2)
        detail = (current[::2] - current[1::2]) / np.sqrt(2)
        coeffs.append(detail)
        current = approx
    coeffs.append(current)
    return coeffs[::-1]


def haar_wavelet_reconstruct(coeffs: list) -> np.ndarray:
    """Haar小波重构（纯numpy实现）"""
    current = coeffs[0].copy()
    for i in range(1, len(coeffs)):
        detail = coeffs[i]
        n = len(current)
        current = np.repeat(current, 2)[:len(detail) * 2]
        # 重构计算
        current[::2] = (current[::2] + detail) / np.sqrt(2)
        current[1::2] = (current[1::2] - detail) / np.sqrt(2)
    return current


def wavelet_denoise_csi(
        raw_mat: np.ndarray,
        level: int = 4,
        threshold_mode: str = 'soft'
) -> np.ndarray:
    """对CSI矩阵进行小波去噪（按子载波独立处理）"""
    T, F = raw_mat.shape
    denoised_mat = np.zeros_like(raw_mat, dtype=np.float32)

    for f in range(F):
        csi_seq = raw_mat[:, f].flatten()
        original_len = len(csi_seq)

        # 小波分解
        coeffs = haar_wavelet_decompose(csi_seq, level=level)
        cA = coeffs[0]  # 近似系数（低频特征）
        cD_list = coeffs[1:]  # 细节系数（高频噪声）

        # 自适应阈值计算
        sigma = np.median(np.abs(cD_list[-1])) / 0.6745  # 稳健估计噪声标准差
        threshold = sigma * np.sqrt(2 * np.log(original_len))  # 通用阈值公式

        # 阈值处理（抑制噪声）
        denoised_cD_list = []
        for cD in cD_list:
            if threshold_mode == 'soft':
                # 软阈值：|x| < t → 0；|x| ≥ t → x ± t
                denoised_cD = np.where(np.abs(cD) < threshold, 0, cD - np.sign(cD) * threshold)
            else:
                # 硬阈值：|x| < t → 0；|x| ≥ t → x
                denoised_cD = np.where(np.abs(cD) < threshold, 0, cD)
            denoised_cD_list.append(denoised_cD)

        # 小波重构
        denoised_coeffs = [cA] + denoised_cD_list
        denoised_seq = haar_wavelet_reconstruct(denoised_coeffs)

        # 长度对齐（确保与原始序列长度一致）
        if len(denoised_seq) > original_len:
            denoised_seq = denoised_seq[:original_len]
        else:
            denoised_seq = np.pad(denoised_seq, (0, original_len - len(denoised_seq)), mode='edge')

        denoised_mat[:, f] = denoised_seq

    return denoised_mat


def load_csi_file(path: str) -> np.ndarray:
    """
    加载单个 CSI 文件为二维矩阵 (T x F)
    支持 .csv / .npy
    返回 float32
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        data = pd.read_csv(path, header=None).values
    elif ext == '.npy':
        data = np.load(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return data.astype(np.float32)


def interpolate_adjust_length(mat: np.ndarray, target_len: int) -> np.ndarray:
    """
    基于插值的长短帧对齐：
    - 短帧（< target_len）：线性插值补齐
    - 长帧（> target_len）：线性插值压缩
    """
    T, F = mat.shape
    if T == target_len:
        return mat

    # 生成原始时间轴和目标时间轴
    original_time = np.linspace(0, 1, T)
    target_time = np.linspace(0, 1, target_len)

    # 对每个子载波单独插值
    adjusted = np.zeros((target_len, F), dtype=mat.dtype)
    for f in range(F):
        interpolator = interp1d(
            original_time, mat[:, f],
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )
        adjusted[:, f] = interpolator(target_time)

    return adjusted


class CSIDataset(Dataset):
    """
    数据处理流程（新增小波去噪步骤）：
      1. 读取原始 (T,F)
      2. 列数校验
      3. 小波去噪（新增步骤）
      4. 基于插值的长短帧对齐
      5. 伪彩色转换（含归一化）
      6. 随机增强 (可选)
      7. 返回 tensor
    """

    def __init__(self,
                 files: List[str],
                 class_to_idx: Dict[str, int],
                 converter: CSIPseudoColorConverter,
                 min_time_len: int,
                 max_time_len: int,
                 subcarriers: int,
                 augment: bool = False,
                 cache: bool = True,
                 # 新增：小波去噪参数
                 wavelet_level: int = 4,
                 wavelet_threshold_mode: str = 'soft'):
        self.files = files
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.converter = converter
        self.min_time_len = min_time_len
        self.max_time_len = max_time_len
        self.target_time_len = (min_time_len + max_time_len) // 2  # 插值目标长度
        self.subcarriers = subcarriers
        self.augment = augment
        self.cache = cache
        self._cache_store = {}

        # 小波去噪参数
        self.wavelet_level = wavelet_level
        self.wavelet_threshold_mode = wavelet_threshold_mode

        # 初始化时完成预处理并缓存
        if self.cache:
            print(f"[INFO] 开始预处理 {len(files)} 个文件（含小波去噪）...")
            for i, f in enumerate(files):
                # 显示进度
                if (i + 1) % 100 == 0:
                    print(f"[INFO] 已预处理 {i + 1}/{len(files)} 个文件")

                try:
                    # 1. 读取原始数据
                    csi = load_csi_file(f)

                    # 2. 列数校验
                    if csi.shape[1] != self.subcarriers:
                        raise ValueError(f"Subcarrier mismatch: {f}, expect {self.subcarriers}, got {csi.shape[1]}")

                    # 3. 新增：小波去噪
                    csi_denoised = wavelet_denoise_csi(
                        csi,
                        level=self.wavelet_level,
                        threshold_mode=self.wavelet_threshold_mode
                    )

                    # 4. 基于插值的长短帧对齐
                    csi_aligned = interpolate_adjust_length(csi_denoised, self.target_time_len)

                    # 5. 伪彩色转换（含归一化）
                    img = self.converter(csi_aligned)

                    # 确保内存布局正确
                    if any(s < 0 for s in img.strides) or (not img.flags['C_CONTIGUOUS']):
                        img = np.ascontiguousarray(img, dtype=np.float32)
                    else:
                        img = img.astype(np.float32, copy=False)

                    # 缓存预处理后的图像
                    self._cache_store[f] = img

                except Exception as e:
                    print(f"[ERROR] 处理文件 {f} 时出错: {str(e)}")
                    continue

    def __len__(self):
        return len(self.files)

    def _get_label(self, path: str) -> int:
        parts = os.path.normpath(path).split(os.sep)
        for p in reversed(parts):
            if p in self.class_to_idx:
                return self.class_to_idx[p]
        raise ValueError(f"无法从路径解析类别: {path}")

    def __getitem__(self, idx):
        fpath = self.files[idx]
        label = self._get_label(fpath)

        # 读取预处理后的数据
        if self.cache and fpath in self._cache_store:
            img = self._cache_store[fpath]
        else:
            # 缓存未命中时的降级流程
            csi = load_csi_file(fpath)
            if csi.shape[1] != self.subcarriers:
                raise ValueError(f"Subcarrier mismatch: {fpath}, expect {self.subcarriers}, got {csi.shape[1]}")

            # 新增：小波去噪（降级流程也需包含）
            csi_denoised = wavelet_denoise_csi(
                csi,
                level=self.wavelet_level,
                threshold_mode=self.wavelet_threshold_mode
            )

            csi_aligned = interpolate_adjust_length(csi_denoised, self.target_time_len)
            img = self.converter(csi_aligned)
            img = np.ascontiguousarray(img, dtype=np.float32)

        # 数据增强
        if self.augment:
            img = random_augment(img)

        # 转换为 torch tensor 并返回
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


def split_dataset(all_files: List[str], train_ratio, val_ratio, test_ratio, seed=42):
    """数据集拆分函数（保持不变）"""
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(all_files))
    rng.shuffle(idxs)
    n = len(all_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return [all_files[i] for i in train_idx], \
        [all_files[i] for i in val_idx], \
        [all_files[i] for i in test_idx]


def collect_files(root: str, classes: List[str], file_ext=".csv"):
    """文件收集函数（保持不变）"""
    res = []
    for cls in classes:
        pattern = os.path.join(root, cls, f"*{file_ext}")
        res.extend(glob.glob(pattern))
    return res