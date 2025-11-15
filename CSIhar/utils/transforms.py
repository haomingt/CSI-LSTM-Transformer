import numpy as np
import matplotlib
import cv2

class CSIPseudoColorConverter:
    """
    将原始 CSI 幅度矩阵 (T x F) 转换为伪彩色图 (C x H x W) (float32 ∈ [0,1])
    流程：
      1. 归一化 (per_sample / global)
      2. colormap -> (T,F,3)
      3. resize 为 (H,W,3)
      4. 转置为 (C,H,W)
    """
    def __init__(self, target_size=(64, 64), cmap='viridis', norm_mode='per_sample'):
        self.target_size = tuple(target_size)
        self.cmap = matplotlib.colormaps.get_cmap(cmap)
        self.norm_mode = norm_mode
        self.global_min = None
        self.global_max = None

    def fit_global(self, matrices_list):
        all_vals = np.concatenate([m.reshape(-1) for m in matrices_list], axis=0)
        self.global_min = all_vals.min()
        self.global_max = all_vals.max()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.norm_mode == 'global':
            assert self.global_min is not None and self.global_max is not None, \
                "必须先调用 fit_global 计算全局 min/max"
            mn, mx = self.global_min, self.global_max
        else:
            mn, mx = x.min(), x.max()

        if mx - mn < 1e-9:  # 常数矩阵
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def __call__(self, csi_matrix: np.ndarray) -> np.ndarray:
        normed = self._normalize(csi_matrix)          # (T,F)
        colored = self.cmap(normed)[:, :, :3]         # (T,F,3)
        img = cv2.resize(colored, self.target_size, interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # (3,H,W)
        return img


def random_augment(img: np.ndarray,
                   hflip_p=0.5,
                   vflip_p=0.0,
                   jitter_p=0.3,
                   contrast_sigma=0.1,
                   brightness_sigma=0.05) -> np.ndarray:
    """
    简单数据增强:
      - 水平翻转 (axis=2)
      - 垂直翻转 (axis=1)
      - 颜色/亮度扰动
    输入:
      img: (C,H,W) float32 ∈ [0,1]
    返回:
      (C,H,W) float32 ∈ [0,1], C_CONTIGUOUS, 无负 stride
    """
    out = img.copy()

    # 水平翻转
    if hflip_p > 0 and np.random.rand() < hflip_p:
        out = out[:, :, ::-1].copy()  # copy 去除负 stride

    # 垂直翻转
    if vflip_p > 0 and np.random.rand() < vflip_p:
        out = out[:, ::-1, :].copy()

    # 颜色/亮度抖动
    if jitter_p > 0 and np.random.rand() < jitter_p:
        alpha = 1.0 + contrast_sigma * np.random.randn()   # 对比度
        beta = brightness_sigma * np.random.randn()        # 亮度
        out = out * alpha + beta
        out = np.clip(out, 0.0, 1.0)

    # 兜底连续 & dtype
    if any(s < 0 for s in out.strides) or (not out.flags['C_CONTIGUOUS']):
        out = np.ascontiguousarray(out, dtype=np.float32)
    else:
        out = out.astype(np.float32, copy=False)

    return out
