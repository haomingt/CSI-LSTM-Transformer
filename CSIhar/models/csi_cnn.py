import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    基础 2D-CNN 模型（对应论文简化结构）：
      Conv -> LeakyReLU -> Pool -> Dropout
      重复若干次后全连接分类
    输入: (B, 3, 64, 64)
    """
    def __init__(self, num_classes=7, in_channels=3, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),       # 64 -> 32
            nn.Dropout(dropout),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),       # 32 -> 16
            nn.Dropout(dropout),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),       # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                         # 128 * 8 * 8
            nn.Linear(128 * 8 * 8, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


class DeeperCNN(nn.Module):
    """
    更深层 CNN：
      - Stem 轻量卷积 + BN
      - 三个下采样 block (每个包含两次卷积 + BN + LeakyReLU + MaxPool)
      - 自适应全局池化 -> 全连接
    """
    def __init__(self, num_classes=7, in_channels=3, dropout=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(2),  # 64->32
        )

        self.layer2 = self._make_block(32, 64)   # 32->16
        self.layer3 = self._make_block(64, 128)  # 16->8
        self.layer4 = self._make_block(128, 256) # 8->4

        # 自适应池化到 (1,1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _make_block(self, in_ch, out_ch):
        """
        标准卷积块：
          Conv + BN + LeakyReLU -> Conv + BN + LeakyReLU -> MaxPool
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


def build_model(name: str, num_classes: int, in_channels=3, dropout=0.25):
    """
    按名称构建模型。
    """
    if name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes, in_channels, dropout)
    elif name.lower() == 'deeper_cnn':
        return DeeperCNN(num_classes, in_channels, dropout)
    else:
        raise ValueError(f"Unknown model name: {name}")
