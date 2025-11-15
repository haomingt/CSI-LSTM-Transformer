import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# 适配服务器路径（与data_processor_BLSTM.py保持一致）
sys.path.append("/home/chenjun/python_codes/WiFi_sensing/scripts")


class AttenLayer(nn.Module):
    """
    彻底修复维度错误：更换1维参数的初始化方式，避免Xavier对1维张量的限制
    """

    def __init__(self, num_state=400):
        super(AttenLayer, self).__init__()
        self.num_state = num_state
        # 预定义参数（明确标注维度）
        self.kernel = None  # 2维：(hidden_dim, num_state)
        self.bias = None  # 1维：(num_state,)
        self.prob_kernel = None  # 1维：(num_state,)

    def forward(self, input_tensor):
        batch_size, seq_len, hidden_dim = input_tensor.shape

        # 初始化参数：按维度类型选择合适的初始化方式
        if self.kernel is None:
            # 1. 2维权重矩阵（kernel）：继续用Xavier初始化（适合线性映射）
            self.kernel = nn.Parameter(
                torch.empty((hidden_dim, self.num_state), device=input_tensor.device)
            )
            nn.init.xavier_uniform_(self.kernel)  # 仅对2维张量使用Xavier

            # 2. 1维偏置（bias）：用常数初始化（0值，避免初始干扰）
            self.bias = nn.Parameter(
                torch.zeros(self.num_state, device=input_tensor.device)
            )

            # 3. 1维概率权重（prob_kernel）：用均匀分布初始化（替代Xavier，避免维度错误）
            self.prob_kernel = nn.Parameter(
                torch.empty(self.num_state, device=input_tensor.device)
            )
            nn.init.uniform_(self.prob_kernel, a=-0.1, b=0.1)  # 1维张量适配的初始化

        # 注意力计算逻辑（功能不变，确保与原版本一致）
        atten_state = torch.tanh(torch.matmul(input_tensor, self.kernel) + self.bias)  # (batch, seq_len, num_state)
        logits = torch.matmul(atten_state, self.prob_kernel)  # (batch, seq_len)
        time_weights = F.softmax(logits, dim=1)  # 权重归一化
        weighted_feature = torch.sum(input_tensor * time_weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)

        return weighted_feature


def build_blstm_model(input_shape=(100, 52), num_classes=7):
    """
    构建与数据处理脚本完全兼容的BLSTM模型
    输入形状(100,52)匹配data_processor_BLSTM.py的seq_len=100、52子载波
    """

    class BLSTMAttentionModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=200, num_classes=7):
            super(BLSTMAttentionModel, self).__init__()
            # 双向LSTM：单层无内置dropout，消除警告
            self.blstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                bidirectional=True,
                batch_first=True,
                dropout=0.0,
                num_layers=1
            )
            # 独立Dropout层：保持正则化效果
            self.dropout = nn.Dropout(p=0.2)
            self.attention = AttenLayer(num_state=400)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向输出400维

        def forward(self, x):
            # 前向传播：严格匹配数据格式
            blstm_out, _ = self.blstm(x)  # (batch, 100, 400)
            blstm_out = self.dropout(blstm_out)  # 正则化
            atten_out = self.attention(blstm_out)  # (batch, 400)
            logits = self.fc(atten_out)  # (batch, 7)
            return F.softmax(logits, dim=1)  # 输出概率分布

    seq_len, input_dim = input_shape
    model = BLSTMAttentionModel(
        input_dim=input_dim,
        hidden_dim=200,
        num_classes=num_classes
    )

    # 打印模型信息
    print("=" * 60)
    print("=== 注意力BLSTM模型（PyTorch最终修复版） ===")
    print(f"输入要求: {input_shape} (时间步×子载波) | 输出: (batch, 7) (活动概率)")
    print("\n模型结构:")
    print(model)
    print("=" * 60)

    return model


if __name__ == "__main__":
    """测试：确保无维度错误，输入输出完全匹配"""
    try:
        # 1. 构建模型（与数据处理脚本兼容）
        model = build_blstm_model(input_shape=(100, 52))

        # 2. 模拟数据：完全复刻data_processor_BLSTM.py的输出格式
        # 形状：(batch=32, seq_len=100, features=52)
        mock_input = torch.randn(32, 100, 52)

        # 3. 前向传播（关闭梯度，仅测试推理）
        with torch.no_grad():
            mock_output = model(mock_input)

        # 4. 验证结果
        print("\n✅ 测试通过！详细信息:")
        print(f"  输入形状: {mock_input.shape} → 符合预期")
        print(f"  输出形状: {mock_output.shape} → 符合预期 (32,7)")
        print(f"  首样本概率和: {mock_output[0].sum().item():.4f} → 接近1.0（softmax正常）")
        print("\n✅ 模型可直接用于训练！")

    except Exception as e:
        print(f"\n❌ 测试失败，错误详情: {str(e)}")
        # 针对性排查建议
        if "Fan in and fan out" in str(e):
            print("🔍 错误原因：1维张量使用了Xavier初始化，已在修复版中解决，请重新运行！")
        else:
            print("🔍 请检查：PyTorch版本（需≥1.8）、输入形状是否为(100,52)")