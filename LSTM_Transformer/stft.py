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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

from src.dataset import CSIDataset
from src.train import train_one_epoch, evaluate
from src.utils import set_seed, collect_files, split_dataset
from src.model_LSTMTransformer import LSTMTransformer

# ===================== 邮箱配置（你自己的） =====================
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465
SENDER_EMAIL = "2825493439@qq.com"
SENDER_PASSWORD = "ozvctjacpnfgdehe"
RECEIVER_EMAIL = "2825493439@qq.com"

# ===================== 全局设置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXCEL_PATH = "ablation_stft_results.xlsx"
CLASSES = ["walk", "run", "sitdown", "standup", "fall", "lie_down", "bend"]

# ===================== 发送邮件 =====================
def send_email(subject, body, attachments):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'utf-8'))

        for fpath in attachments:
            if os.path.exists(fpath):
                part = MIMEBase('application', 'octet-stream')
                with open(fpath, 'rb') as f:
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(fpath)}"')
                msg.attach(part)

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("📩 邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败：{e}")

# ===================== 保存实验表格 =====================
def save_excel(results):
    rows = []
    for exp_name, res in results.items():
        row = {
            "实验": exp_name,
            "总体准确率": round(res["acc"], 4),
            "宏观精确率": round(res["macro_p"], 4),
            "宏观召回率": round(res["macro_r"], 4),
            "宏观F1": round(res["macro_f1"], 4),
        }
        for i, cls in enumerate(CLASSES):
            row[f"{cls}_精确率"] = round(res["class_p"][i], 4)
            row[f"{cls}_召回率"] = round(res["class_r"][i], 4)
            row[f"{cls}_F1"] = round(res["class_f1"][i], 4)
        rows.append(row)
    pd.DataFrame(rows).to_excel(EXCEL_PATH, index=False)

# ===================== 运行一组实验 =====================
def run_exp(cfg, use_stft):
    set_seed(cfg["seed"])
    exp_name = "with_STFT" if use_stft else "no_STFT"
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)

    # 数据加载
    grouped_files = collect_files(cfg["data"]["raw_root"], CLASSES, cfg["data"]["file_ext"])
    train_f, val_f, test_f = split_dataset(
        grouped_files,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
        cfg["data"]["test_split"],
        cfg["seed"]
    )

    # 数据集
    ds_args = dict(
        class_to_idx={c:i for i,c in enumerate(CLASSES)},
        max_time_len=cfg["data"]["max_time_len"],
        min_time_len=cfg["data"]["min_time_len"],
        subcarriers=cfg["data"]["subcarriers"],
        cache=cfg["data"]["cache_in_memory"],
        use_wavelet=cfg["data"]["use_wavelet"],
        wavelet_level=cfg["data"]["wavelet_level"],
        wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
        use_stft=use_stft
    )
    train_ds = CSIDataset(train_f, **ds_args, augment=True)
    val_ds = CSIDataset(val_f, **ds_args, augment=False)
    test_ds = CSIDataset(test_f, **ds_args, augment=False)

    # 加载器
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # 模型
    model = LSTMTransformer(
        input_dim=train_ds.feature_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        num_classes=7
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=float(cfg["training"]["weight_decay"]))
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["training"]["amp"])
    warmup = cfg["training"]["warmup_epochs"]

    # 训练
    best_acc = 0
    patience = 0
    model_path = os.path.join(out_dir, "best.pth")

    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, DEVICE, criterion, optimizer, scaler, epoch, warmup)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, DEVICE, criterion)

        print(f"[{exp_name}] Epoch {epoch} | train={train_acc:.2f}% | val={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience +=1
            if patience >= cfg["training"]["early_stop_patience"]:
                print("🛑 早停")
                break

    # 测试
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, y_pred, y_true, report = evaluate(
        model, test_loader, DEVICE, criterion,
        classes=CLASSES, save_path=out_dir
    )

    # 计算指标
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    class_p, class_r, class_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(7), zero_division=0)

    res = {
        "acc": test_acc/100,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "class_p": class_p,
        "class_r": class_r,
        "class_f1": class_f1,
        "cm": os.path.join(out_dir, "confusion_matrix.png")
    }
    return res

# ===================== 主函数 =====================
def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results = {}
    print("\n========== 实验1：使用 STFT ==========")
    results["with_STFT"] = run_exp(cfg, use_stft=True)

    print("\n========== 实验2：不使用 STFT ==========")
    results["no_STFT"] = run_exp(cfg, use_stft=False)

    # 保存表格
    save_excel(results)

    # 发邮件
    send_email(
        subject="【毕设】STFT消融实验全部完成",
        body="有无STFT对比结果已生成，包含表格+混淆矩阵",
        attachments=[EXCEL_PATH, results["with_STFT"]["cm"], results["no_STFT"]["cm"]]
    )

if __name__ == "__main__":
    main()