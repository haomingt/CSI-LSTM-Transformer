import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.signal import stft
from .utils import haar_wavelet_decompose, haar_wavelet_reconstruct
from scipy.interpolate import interp1d

def load_csi_file(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        data = pd.read_csv(path, header=None).values
    elif ext == '.npy':
        data = np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return data.astype(np.float32)

def interpolate_adjust_length(mat: np.ndarray, target_len: int) -> np.ndarray:
    T, F = mat.shape
    if T == target_len:
        return mat
    original_time = np.linspace(0, 1, T)
    target_time = np.linspace(0, 1, target_len)
    adjusted = np.zeros((target_len, F), dtype=mat.dtype)
    for f in range(F):
        interpolator = interp1d(original_time, mat[:, f], kind='linear', bounds_error=False, fill_value="extrapolate")
        adjusted[:, f] = interpolator(target_time)
    return adjusted

def wavelet_denoise_csi(raw_mat: np.ndarray, level: int = 4, threshold_mode: str = 'soft') -> np.ndarray:
    T, F = raw_mat.shape
    denoised = np.zeros_like(raw_mat)
    for f in range(F):
        coeffs = haar_wavelet_decompose(raw_mat[:, f], level)
        cA = coeffs[0]
        cD_list = coeffs[1:]
        threshold = np.median(np.abs(cD_list[-1])) * 2
        denoised_cD = []
        for cd in cD_list:
            if threshold_mode == 'soft':
                denoised_cd = np.sign(cd) * np.maximum(np.abs(cd) - threshold, 0)
            else:
                denoised_cd = np.where(np.abs(cd) < threshold, 0, cd)
            denoised_cD.append(denoised_cd)
        denoised[:, f] = haar_wavelet_reconstruct([cA] + denoised_cD)[:T]
    return denoised

def apply_stft(csi: np.ndarray, nperseg, noverlap, reduce=True) -> np.ndarray:
    T, F = csi.shape
    stft_result = []
    for f in range(F):
        _, _, Zxx = stft(csi[:, f], nperseg=nperseg, noverlap=noverlap)
        Zxx = np.abs(Zxx)
        if reduce:
            Zxx = Zxx[:Zxx.shape[0]//2, :]
        stft_result.append(Zxx)
    stft_result = np.stack(stft_result, axis=-1)
    time_steps = stft_result.shape[1]
    stft_result = stft_result.transpose(1, 0, 2).reshape(time_steps, -1)
    return stft_result.astype(np.float32)

class CSIDataset(Dataset):
    def __init__(self, 
                  
                 files: list, class_to_idx: dict,
                 use_wavelet: bool,
                 max_time_len: int, min_time_len: int,
                 subcarriers: int, augment: bool = False,
                 cache: bool = True,
                 
                 wavelet_level: int = 4,
                 wavelet_threshold_mode: str = 'soft',
                 use_stft: bool = True,
                 stft_reduce: bool = True,
                 nperseg=2, 
                 noverlap=1,
                ):
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
        self.use_stft = use_stft
        self.stft_reduce = stft_reduce
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.feature_dim = None
        if self.cache:
            self._preprocess_all()

    def _preprocess_all(self):
        for f in self.files:
            try:
                csi = load_csi_file(f)
                if csi.shape[1] != self.subcarriers:
                    raise ValueError(f"Subcarriers mismatch in file {f}")
                if self.use_wavelet:
                    csi = wavelet_denoise_csi(csi, level=self.wavelet_level, threshold_mode=self.wavelet_threshold_mode)
                csi = (csi - csi.mean()) / (csi.std() + 1e-8)
                if self.use_stft:
                    csi = apply_stft(csi, self.nperseg,self.noverlap,reduce=self.stft_reduce)
                csi = interpolate_adjust_length(csi, self.target_time_len)
                self._cache[f] = csi
            except Exception as e:
                print(f"Error processing {f}: {e}")
        if self._cache:
            self.feature_dim = next(iter(self._cache.values())).shape[-1]

    def _get_label(self, path: str) -> int:
        parts = os.path.normpath(path).split(os.sep)
        for p in reversed(parts):
            if p in self.class_to_idx:
                return self.class_to_idx[p]
        raise ValueError(f"Cannot find label in path {path}")

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
                csi = wavelet_denoise_csi(csi, level=self.wavelet_level, threshold_mode=self.wavelet_threshold_mode)
            csi = (csi - csi.mean()) / (csi.std() + 1e-8)
            if self.use_stft:
                csi = apply_stft(csi, self.nperseg,self.noverlap,reduce=self.stft_reduce)
            csi = interpolate_adjust_length(csi, self.target_time_len)
        return torch.from_numpy(csi).float(), torch.tensor(label, dtype=torch.long)