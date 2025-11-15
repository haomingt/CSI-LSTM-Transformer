import os
import glob
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils.transforms import CSIPseudoColorConverter, random_augment


# 频域特征提取函数
def time_to_freq_domain(mat: np.ndarray, freq_bins: int = 64) -> np.ndarray:
    """将时域信号转换为频域特征"""
    T, F = mat.shape
    # 打印输入时域数据的形状（调试信息）
    print(f"[频域变换] 输入时域数据形状: {mat.shape} (时间步长: {T}, 子载波: {F})")

    freq_mat = np.zeros((freq_bins, F), dtype=np.float32)

    for f in range(F):
        time_signal = mat[:, f].flatten()
        fft_result = np.fft.fft(time_signal)
        magnitude = np.abs(fft_result)
        magnitude = np.log1p(magnitude)  # 对数变换增强小信号
        half_magnitude = magnitude[:len(magnitude) // 2]  # 取频谱前半部分（对称特性）

        # 插值调整到目标频率点数
        if len(half_magnitude) < freq_bins:
            freq_axis = np.linspace(0, 1, len(half_magnitude))
            target_axis = np.linspace(0, 1, freq_bins)
            magnitude_resized = np.interp(target_axis, freq_axis, half_magnitude)
        else:
            magnitude_resized = half_magnitude[:freq_bins]

        freq_mat[:, f] = magnitude_resized

    # 打印输出频域数据的形状（验证变换结果）
    print(f"[频域变换] 输出频域数据形状: {freq_mat.shape} (频率 bins: {freq_bins}, 子载波: {F})")
    return freq_mat


# 小波去噪相关函数
def haar_wavelet_decompose(signal: np.ndarray, level: int) -> list:
    """使用Haar小波进行信号分解"""
    coeffs = []
    current = signal.copy()
    for _ in range(level):
        if len(current) % 2 != 0:
            current = np.append(current, current[-1])  # 确保偶数长度
        approx = (current[::2] + current[1::2]) / np.sqrt(2)  # 近似系数
        detail = (current[::2] - current[1::2]) / np.sqrt(2)  # 细节系数
        coeffs.append(detail)
        current = approx
    coeffs.append(current)  # 最后一层近似系数
    return coeffs[::-1]  # 反转顺序，[近似系数, 细节系数1, 细节系数2, ...]


def haar_wavelet_reconstruct(coeffs: list) -> np.ndarray:
    """从小波系数重构信号"""
    current = coeffs[0].copy()  # 近似系数
    for i in range(1, len(coeffs)):
        detail = coeffs[i]
        n = len(current)
        # 扩展近似系数长度
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
    """对CSI矩阵进行小波去噪"""
    T, F = raw_mat.shape
    denoised_mat = np.zeros_like(raw_mat, dtype=np.float32)
    for f in range(F):
        csi_seq = raw_mat[:, f].flatten()
        original_len = len(csi_seq)
        # 小波分解
        coeffs = haar_wavelet_decompose(csi_seq, level=level)
        cA = coeffs[0]  # 近似系数
        cD_list = coeffs[1:]  # 细节系数列表

        # 计算阈值（基于最后一层细节系数的中值）
        sigma = np.median(np.abs(cD_list[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(original_len))

        # 对细节系数进行阈值处理
        denoised_cD_list = []
        for cD in cD_list:
            if threshold_mode == 'soft':
                denoised_cD = np.where(np.abs(cD) < threshold, 0, cD - np.sign(cD) * threshold)
            else:  # 硬阈值
                denoised_cD = np.where(np.abs(cD) < threshold, 0, cD)
            denoised_cD_list.append(denoised_cD)

        # 重构信号
        denoised_coeffs = [cA] + denoised_cD_list
        denoised_seq = haar_wavelet_reconstruct(denoised_coeffs)

        # 调整长度与原始一致
        if len(denoised_seq) > original_len:
            denoised_seq = denoised_seq[:original_len]
        else:
            denoised_seq = np.pad(denoised_seq, (0, original_len - len(denoised_seq)), mode='edge')
        denoised_mat[:, f] = denoised_seq
    return denoised_mat


def zscore_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    对CSI矩阵进行zscore标准化
    z = (x - μ) / σ，其中μ为均值，σ为标准差
    eps用于避免除以零的情况
    """
    mean = np.mean(mat, axis=0, keepdims=True)  # 按子载波计算均值
    std = np.std(mat, axis=0, keepdims=True)  # 按子载波计算标准差
    return (mat - mean) / (std + eps)


# 数据加载工具函数
def load_csi_file(path: str) -> np.ndarray:
    """加载CSI数据文件（支持csv和npy格式）"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        data = pd.read_csv(path, header=None).values
    elif ext == '.npy':
        data = np.load(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    return data.astype(np.float32)


def interpolate_adjust_length(mat: np.ndarray, target_len: int) -> np.ndarray:
    """通过插值调整CSI矩阵的时间长度"""
    T, F = mat.shape
    if T == target_len:
        return mat
    original_time = np.linspace(0, 1, T)
    target_time = np.linspace(0, 1, target_len)
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


# 核心数据集类
class CSIDataset(Dataset):
    def __init__(self,
                 files: List[str],
                 class_to_idx: Dict[str, int],
                 converter: CSIPseudoColorConverter,
                 min_time_len: int,
                 max_time_len: int,
                 subcarriers: int,
                 augment: bool = False,
                 cache: bool = True,
                 wavelet_level: int = 4,
                 wavelet_threshold_mode: str = 'soft',
                 use_wavelet: bool = False,  # 添加小波开关参数
                 use_zscore: bool = True):
        self.files = files
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.converter = converter
        self.min_time_len = min_time_len
        self.max_time_len = max_time_len
        self.target_time_len = (min_time_len + max_time_len) // 2  # 目标时间长度（取平均）
        self.subcarriers = subcarriers  # 子载波数量
        self.augment = augment  # 是否启用数据增强
        self.cache = cache  # 是否缓存预处理结果
        self._cache_store = {}  # 缓存存储
        self.freq_bins = 64  # 频域特征点数（与图像高度保持一致）
        self.wavelet_level = wavelet_level  # 小波分解层数
        self.wavelet_threshold_mode = wavelet_threshold_mode  # 小波阈值模式
        self.use_wavelet = use_wavelet  # 小波变换开关（新增）
        self.use_zscore = use_zscore  # 是否使用zscore标准化

        # 预处理并缓存数据
        if self.cache:
            print(f"[INFO] 开始预处理 {len(files)} 个文件（小波去噪+zscore标准化+频域转换）...")
            for i, f in enumerate(files):
                if (i + 1) % 100 == 0:
                    print(f"[INFO] 已预处理 {i + 1}/{len(files)} 个文件")
                try:
                    # 加载原始时域数据
                    csi_time = load_csi_file(f)
                    if csi_time.shape[1] != self.subcarriers:
                        raise ValueError(f"子载波数量不匹配: {f}, 预期 {self.subcarriers}, 实际 {csi_time.shape[1]}")

                    if self.use_wavelet:  # 使用配置的开关
                        csi_denoised = wavelet_denoise_csi(
                            csi_time,
                            level=self.wavelet_level,
                            threshold_mode=self.wavelet_threshold_mode
                        )
                    else:
                        csi_denoised = csi_time  # 不使用小波变换，直接使用原始数据

                    # zscore标准化步骤
                    if self.use_zscore:
                        csi_denoised = zscore_normalize(csi_denoised)

                    # 转换到频域
                    print(f"[文件 {os.path.basename(f)}] 开始频域变换")
                    csi_freq = time_to_freq_domain(
                        csi_denoised,
                        freq_bins=self.freq_bins
                    )

                    # 转换为伪彩色图像
                    img = self.converter(csi_freq)
                    # 确保内存连续性和数据类型
                    if any(s < 0 for s in img.strides) or (not img.flags['C_CONTIGUOUS']):
                        img = np.ascontiguousarray(img, dtype=np.float32)
                    else:
                        img = img.astype(np.float32, copy=False)
                    self._cache_store[f] = img
                except Exception as e:
                    print(f"[ERROR] 处理文件 {f} 时出错: {str(e)}")
                    continue

    def __len__(self):
        return len(self.files)

    def _get_label(self, path: str) -> int:
        """从文件路径解析类别标签"""
        parts = os.path.normpath(path).split(os.sep)
        for p in reversed(parts):
            if p in self.class_to_idx:
                return self.class_to_idx[p]
        raise ValueError(f"无法从路径解析类别: {path}")

    def __getitem__(self, idx):
        fpath = self.files[idx]
        label = self._get_label(fpath)

        # 从缓存加载或实时处理
        if self.cache and fpath in self._cache_store:
            img = self._cache_store[fpath]
        else:
            csi_time = load_csi_file(fpath)
            if csi_time.shape[1] != self.subcarriers:
                raise ValueError(f"子载波数量不匹配: {fpath}")

            # 根据开关决定是否应用小波去噪
            if self.use_wavelet:
                csi_denoised = wavelet_denoise_csi(
                    csi_time,
                    level=self.wavelet_level,
                    threshold_mode=self.wavelet_threshold_mode
                )
            else:
                csi_denoised = csi_time

            # zscore标准化步骤
            if self.use_zscore:
                csi_denoised = zscore_normalize(csi_denoised)

            # 频域转换
            csi_freq = time_to_freq_domain(csi_denoised, freq_bins=self.freq_bins)
            # 伪彩色转换
            img = self.converter(csi_freq)
            img = np.ascontiguousarray(img, dtype=np.float32)

        # 数据增强
        if self.augment:
            img = random_augment(img)

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# 数据集拆分函数
def split_dataset(all_files: List[str], train_ratio, val_ratio, test_ratio, seed=42):
    """按比例拆分数据集为训练集、验证集和测试集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6, "比例总和必须为1"
    rng = np.random.RandomState(seed)  # 固定随机种子确保可复现
    idxs = np.arange(len(all_files))
    rng.shuffle(idxs)
    n = len(all_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # 划分索引
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return [all_files[i] for i in train_idx], \
        [all_files[i] for i in val_idx], \
        [all_files[i] for i in test_idx]


def collect_files(root: str, classes: List[str], file_ext=".csv"):
    """收集指定类别下的所有数据文件"""
    res = []
    for cls in classes:
        pattern = os.path.join(root, cls, f"*{file_ext}")
        res.extend(glob.glob(pattern))
    return res