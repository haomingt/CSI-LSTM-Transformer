import os
import yaml
import torch
import numpy as np
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score
import warnings
warnings.filterwarnings('ignore')

from src.dataset import CSIDataset
from src.model_LSTMTransformer import LSTMTransformer
from src.train import evaluate
from src.utils import set_seed, collect_files, split_dataset

# ===================== 邮箱配置 =====================
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465
SENDER_EMAIL = "2825493439@qq.com"
SENDER_PASSWORD = "ozvctjacpnfgdehe"
RECEIVER_EMAIL = "2825493439@qq.com"

# ===================== 显卡 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 你要的 20组 经典STFT参数 =====================
STFT_PARAMS = [
    (8, 4),
    (10, 5),
    (16, 8),
    (24, 12),
    (32, 16),
    (48, 24),
    (64, 32),
    (72, 36),
    (96, 48),
    (128, 64),
    (4, 1),
    (4, 2),
    (8, 2),
    (10, 3),
    (12, 4),
    (12, 6),
    (16, 4),
    (20, 5),
    (20, 10),
    (64, 16),
]

EXCEL_PATH = "stft_ablation_result.xlsx"
EPOCHS = 1000        # 睡觉狂跑1000轮
PATIENCE = 30        # 30轮不提升自动停

# 启动清空旧表
if os.path.exists(EXCEL_PATH):
    os.remove(EXCEL_PATH)

def send_email_with_attachment(subject, body, attachment_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        part = MIMEBase('application', 'octet-stream')
        with open(attachment_path, 'rb') as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
        msg.attach(part)

        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("📩 邮件已发送！")
    except Exception as e:
        print("邮件发送失败", e)

def save_result_to_excel(results, excel_path):
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)

def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    classes = cfg["data"]["classes"]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    grouped_files = collect_files(cfg["data"]["raw_root"], classes, cfg["data"]["file_ext"])
    train_files, val_files, test_files = split_dataset(
        grouped_files,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
        cfg["data"]["test_split"],
        cfg["seed"]
    )

    results = []

    for idx, (nseg, novl) in enumerate(STFT_PARAMS, 1):
        print(f"\n==================================================")
        print(f"🚀 第 {idx}/{len(STFT_PARAMS)} 组 | STFT = ({nseg}, {novl})")

        # ✅ 完全按你的格式修复，不报错
        train_ds = CSIDataset(
            train_files, class_to_idx,
            max_time_len=cfg["data"]["max_time_len"],
            min_time_len=cfg["data"]["min_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
            augment=True,
            cache=cfg["data"]["cache_in_memory"],
            use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=True,
            nperseg=nseg,
            noverlap=novl
        )

        val_ds = CSIDataset(
            val_files, class_to_idx,
            max_time_len=cfg["data"]["max_time_len"],
            min_time_len=cfg["data"]["min_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
            augment=False,
            cache=cfg["data"]["cache_in_memory"],
            use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=True,
            nperseg=nseg,
            noverlap=novl
        )

        test_ds = CSIDataset(
            test_files, class_to_idx,
            max_time_len=cfg["data"]["max_time_len"],
            min_time_len=cfg["data"]["min_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
            augment=False,
            cache=cfg["data"]["cache_in_memory"],
            use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=True,
            nperseg=nseg,
            noverlap=novl
        )

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        model = LSTMTransformer(
            input_dim=train_ds.feature_dim,
            hidden_dim=cfg["models"]["lstm_transformer"]["hidden_dim"],
            num_heads=cfg["models"]["lstm_transformer"]["num_heads"],
            num_layers=cfg["models"]["lstm_transformer"]["num_layers"],
            num_classes=len(classes),
            dropout=cfg["models"]["lstm_transformer"]["dropout"]
        )

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        patience = 0

        for epoch in range(EPOCHS):
            model.train()
            for data, lab in train_loader:
                data, lab = data.to(DEVICE), lab.to(DEVICE)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, lab)
                loss.backward()
                optimizer.step()

            # 验证
            v_loss, v_acc, _, _, _ = evaluate(model, val_loader, DEVICE, criterion)
            print(f"Epoch {epoch+1:>3} | Val Acc: {v_acc:.4f}")

            if v_acc > best_acc:
                best_acc = v_acc
                patience = 0
                torch.save(model.state_dict(), f"best_model_{nseg}_{novl}.pth")
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"\n⏸️ 连续 {PATIENCE} 轮无提升，自动停止！")
                    break

        # 测试
        t_loss, t_acc, y_true, y_pred, _ = evaluate(model, test_loader, DEVICE, criterion)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)

        row = {
            "nperseg": nseg,
            "noverlap": novl,
            "Acc": round(t_acc, 4),
            "walk": round(recall[0], 4),
            "run": round(recall[1], 4),
            "sitdown": round(recall[2], 4),
            "standup": round(recall[3], 4),
            "fall": round(recall[4], 4),
            "lie_down": round(recall[5], 4),
            "bend": round(recall[6], 4)
        }
        results.append(row)
        save_result_to_excel(results, EXCEL_PATH)

        send_email_with_attachment(
            f"【毕设进度】第{idx}组完成 STFT({nseg},{novl})",
            f"总精度 Acc = {t_acc:.4f}\n已自动保存Excel",
            EXCEL_PATH
        )

    send_email_with_attachment("✅ 全部20组STFT实验完成！", "快去写论文！", EXCEL_PATH)
    print("\n🎉 所有实验全部跑完！！！")

if __name__ == "__main__":
    main()