import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from .utils import haar_wavelet_decompose, haar_wavelet_reconstruct


def load_csi_file(path: str) -> np.ndarray:
    """加载CSI文件，支持csv和npy格式"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        data = pd.read_csv(path, header=None).values
    elif ext == '.npy':
        data = np.load(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    return data.astype(np.float32)


def interpolate_adjust_length(mat: np.ndarray, target_len: int) -> np.ndarray:
    """调整时间序列长度，使用插值方法保持数据分布"""
    T, F = mat.shape
    if T == target_len:
        return mat
    original_time = np.linspace(0, 1, T)
    target_time = np.linspace(0, 1, target_len)
    adjusted = np.zeros((target_len, F), dtype=mat.dtype)
    for f in range(F):
        from scipy.interpolate import interp1d
        interpolator = interp1d(
            original_time, mat[:, f],
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )
        adjusted[:, f] = interpolator(target_time)
    return adjusted


def wavelet_denoise_csi(raw_mat: np.ndarray, level: int = 4, threshold_mode: str = 'soft') -> np.ndarray:
    """对CSI数据进行小波去噪处理"""
    T, F = raw_mat.shape
    denoised = np.zeros_like(raw_mat)
    for f in range(F):
        coeffs = haar_wavelet_decompose(raw_mat[:, f], level)
        cA = coeffs[0]
        cD_list = coeffs[1:]

        # 计算阈值
        threshold = np.median(np.abs(cD_list[-1])) * 2

        # 应用阈值
        denoised_cD = []
        for cd in cD_list:
            if threshold_mode == 'soft':
                denoised_cd = np.sign(cd) * np.maximum(np.abs(cd) - threshold, 0)
            else:  # hard
                denoised_cd = np.where(np.abs(cd) < threshold, 0, cd)
            denoised_cD.append(denoised_cd)

        denoised[:, f] = haar_wavelet_reconstruct([cA] + denoised_cD)[:T]
    return denoised


class CSIDataset(Dataset):
    def __init__(self, files: list, class_to_idx: dict,
                 max_time_len: int, min_time_len: int,
                 subcarriers: int, augment: bool = False,
                 cache: bool = True,
                 use_wavelet: bool = True,
                 wavelet_level: int = 4,
                 wavelet_threshold_mode: str = 'soft'):
        self.files = files
        self.class_to_idx = class_to_idx
        self.target_time_len = (max_time_len + min_time_len) // 2
        self.subcarriers = subcarriers
        self.augment = augment
        self.cache = cache
        self._cache = {}

        self.use_wavelet = use_wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_threshold_mode = wavelet_threshold_mode

        if self.cache:
            self._preprocess_all()

    def _preprocess_all(self):
        """预处理所有数据并缓存"""
        for f in self.files:
            try:
                csi = load_csi_file(f)
                if csi.shape[1] != self.subcarriers:
                    raise ValueError(
                        f"子载波数量不匹配在文件 {f}: 预期 {self.subcarriers}, 实际 {csi.shape[1]}"
                    )

                if self.use_wavelet:
                    csi = wavelet_denoise_csi(
                        csi,
                        level=self.wavelet_level,
                        threshold_mode=self.wavelet_threshold_mode
                    )

                # 幅值归一化
                csi = (csi - csi.mean()) / (csi.std() + 1e-8)

                csi_aligned = interpolate_adjust_length(csi, self.target_time_len)
                self._cache[f] = csi_aligned
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")

    def _get_label(self, path: str) -> int:
        """从文件路径中提取标签"""
        parts = os.path.normpath(path).split(os.sep)
        for p in reversed(parts):
            if p in self.class_to_idx:
                return self.class_to_idx[p]
        raise ValueError(f"无法从路径 {path} 中找到标签")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        label = self._get_label(fpath)

        if self.cache and fpath in self._cache:
            csi = self._cache[fpath]
        else:
            csi = load_csi_file(fpath)
            if self.use_wavelet:
                csi = wavelet_denoise_csi(
                    csi,
                    level=self.wavelet_level,
                    threshold_mode=self.wavelet_threshold_mode
                )
            csi = (csi - csi.mean()) / (csi.std() + 1e-8)
            csi = interpolate_adjust_length(csi, self.target_time_len)

        # 数据增强
        if self.augment:
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.9, 1.1)
                csi = interpolate_adjust_length(csi, int(self.target_time_len * scale))
                csi = interpolate_adjust_length(csi, self.target_time_len)
            if np.random.random() < 0.2:
                noise_level = np.random.uniform(0.005, 0.02)
                csi += np.random.normal(0, noise_level, csi.shape)

        return torch.from_numpy(csi).float(), torch.tensor(label, dtype=torch.long)
