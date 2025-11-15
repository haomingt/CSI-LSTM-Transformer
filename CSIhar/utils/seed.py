import os
import random
import numpy as np
import torch

def set_seed(seed: int = 2024):
    """
    设定 Python / NumPy / PyTorch 随机种子，确保结果可复现。
    注意：设置 cudnn.deterministic=True 会降低某些操作性能。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保卷积等操作确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
