import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import time

# 你的模型
from src.model_LSTM import LSTMModel
from src.model_LSTMGMP import LSTM_GMP
from src.model_LSTMTransformer import LSTMTransformer

# 数据集
from src.dataset import CSIDataset
from torch.utils.data import DataLoader

# ====================== 加载配置 ======================
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ====================== 邮箱 ======================
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465
SENDER_EMAIL = "2825493439@qq.com"
SENDER_PASSWORD = "ozvctjacpnfgdehe"
RECEIVER_EMAIL = "2825493439@qq.com"

# ====================== 实验参数 ======================
USE_WAVELET = True
USE_STFT = True
BEST_STFT = (2, 1)
EPOCHS = 20
BATCH_SIZE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 训练 & 测试 ======================
def train_and_evaluate(model, train_loader, test_loader):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(p.numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return round(acc,4), round(prec,4), round(rec,4), round(f1,4)

# ====================== 发邮件 ======================
def send_email(df):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = "【毕业设计】多模型对比结果（小波+STFT）"

    text = f"""
毕业设计：多模型行为识别对比实验
实验配置：
• 小波变换：开启
• STFT参数：nperseg=2, noverlap=1
• 批次大小：2
• 训练轮数：20

实验指标（准确率/精确率/召回率/F1）：
{df.to_string(index=False)}

数据可直接用于毕业论文！
"""
    msg.attach(MIMEText(text, "plain", "utf-8"))

    df.to_excel("model_all_metrics.xlsx", index=False)

    with open("model_all_metrics.xlsx", "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment; filename=model_all_metrics.xlsx")
    msg.attach(part)

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print("✅ 结果已发送到你的QQ邮箱！")

# ====================== 主程序 ======================
if __name__ == "__main__":
    # ==============================================
    # 这部分是你自己项目里的数据集路径
    # ==============================================
    data_cfg = cfg["data"]
    train_files = data_cfg["train_files"]
    val_files = data_cfg["val_files"]
    test_files = data_cfg["test_files"]
    class_to_idx = data_cfg["class_to_idx"]

    # 数据集（完全按你的写法）
    train_ds = CSIDataset(
        train_files, class_to_idx,
        use_wavelet=True,
        min_time_len=cfg["data"]["min_time_len"],
        max_time_len=cfg["data"]["max_time_len"],
        subcarriers=cfg["data"]["subcarriers"],
        use_stft=USE_STFT,
        nperseg=BEST_STFT[0], noverlap=BEST_STFT[1]
    )

    test_ds = CSIDataset(
        test_files, class_to_idx,
        use_wavelet=True,
        min_time_len=cfg["data"]["min_time_len"],
        max_time_len=cfg["data"]["max_time_len"],
        subcarriers=cfg["data"]["subcarriers"],
        use_stft=USE_STFT,
        nperseg=BEST_STFT[0], noverlap=BEST_STFT[1]
    )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    # 模型
    model_list = {
        "LSTM": LSTMModel(),
        "LSTMGMP": LSTM_GMP(),
        "LSTMTransformer": LSTMTransformer(),
    }

    results = []
    for name, model in model_list.items():
        print(f"\n🚀 正在训练：{name}")
        acc, prec, rec, f1 = train_and_evaluate(model, train_loader, test_loader)
        results.append([name, acc, prec, rec, f1])
        print(f"✅ 结果 → Acc={acc}  Prec={prec}  Recall={rec}  F1={f1}")

    df = pd.DataFrame(results, columns=["模型", "准确率", "精确率", "召回率", "F1分数"])
    df = df.sort_values(by="准确率", ascending=False)

    print("\n" + "="*60)
    print("📊 最终排名：")
    print(df.to_string(index=False))

    send_email(df)
    print("\n🎉 全部完成！已发送至邮箱！")