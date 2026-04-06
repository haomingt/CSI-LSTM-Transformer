import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,     # 特征维度，比如 52
        hidden_size: int,    # LSTM 隐藏层
        num_layers: int,     # LSTM 层数
        num_classes: int,    # 分类数（7）
        dropout: float = 0.5,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(
            hidden_size * self.num_directions,
            num_classes
        )

    def forward(self, x):
        """
        x: [B, T, C]
        """
        lstm_out, _ = self.lstm(x)
        # lstm_out: [B, T, H * directions]

        # 方式 1：取最后一个时间步
        out = lstm_out[:, -1, :]

        # 方式 2（可选）：时间平均池化
        # out = lstm_out.mean(dim=1)

        logits = self.fc(out)
        return logits