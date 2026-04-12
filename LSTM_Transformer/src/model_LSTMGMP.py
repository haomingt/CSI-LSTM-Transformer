import torch
import torch.nn as nn

class LSTM_GMP(nn.Module):
    """
    LSTM + Global Max Pooling (GMP) for CSI-based HAR
    Input shape : [B, T, F]
    Output      : [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_dim * self.num_directions,
            num_classes
        )

    def forward(self, x):
        """
        x: [B, T, F]
        """
        # LSTM output: [B, T, hidden*dir]
        lstm_out, _ = self.lstm(x)

        # ===================== 核心改动：GMP =====================
        # 全局最大池化：在时间维度上取最大值
        # lstm_out shape: [B, T, hidden_dim * dir]
        out = torch.max(lstm_out, dim=1).values  # [B, hidden*dir]
        # =========================================================

        out = self.dropout(out)
        logits = self.classifier(out)

        return logits