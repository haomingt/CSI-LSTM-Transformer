import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

from src.dataset import CSIDataset
from src.model_LSTMTransformer import LSTMTransformer
from src.utils import set_seed, collect_files, split_dataset

# ===================== 邮箱 =====================
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465
SENDER_EMAIL = "2825493439@qq.com"
SENDER_PASSWORD = "ozvctjacpnfgdehe"
RECEIVER_EMAIL = "2825493439@qq.com"

# ===================== 显卡 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 实验配置 =====================
EXCEL_PATH = "ablation_stft_final.xlsx"
EPOCHS = 1000
PATIENCE = 800
BEST_STFT = (2, 1)  # 你默认的最优STFT参数

if os.path.exists(EXCEL_PATH):
    os.remove(EXCEL_PATH)

# ===================== 评估 =====================
def evaluate_full(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, lab in loader:
            data, lab = data.to(device), lab.to(device)
            out = model(data)
            pred = torch.argmax(out, dim=1)
            y_true.append(lab.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    return 0.0, acc, y_true, y_pred

# ===================== 混淆矩阵 =====================
def plot_cm(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ===================== 发邮件 =====================
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
        print("📩 邮件发送成功")
    except:
        print("邮件发送失败")

# ===================== 保存Excel =====================
def save_result_to_excel(results, excel_path):
    pd.DataFrame(results).to_excel(excel_path, index=False)

# ===================== 运行一组实验 =====================
def run_exp(cfg, use_stft):
    set_seed(cfg["seed"])
    classes = cfg["data"]["classes"]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    grouped_files = collect_files(cfg["data"]["raw_root"], classes, cfg["data"]["file_ext"])
    train_files, val_files, test_files = split_dataset(grouped_files, 0.7, 0.15, 0.15)

    exp_name = "使用STFT" if use_stft else "不使用STFT"
    print(f"\n======================================")
    print(f"🚀 开始实验：{exp_name}")
    print(f"======================================")

    BEST_MODEL_PATH = f"best_{'stft' if use_stft else 'nostft'}.pth"

    # 数据集（完全和你最优代码一致）
    train_ds = CSIDataset(train_files, class_to_idx,
        min_time_len=cfg["data"]["min_time_len"],
        max_time_len=cfg["data"]["max_time_len"],
        subcarriers=cfg["data"]["subcarriers"],
        use_stft=use_stft,
        nperseg=BEST_STFT[0], noverlap=BEST_STFT[1])

    val_ds = CSIDataset(val_files, class_to_idx,
        min_time_len=cfg["data"]["min_time_len"],
        max_time_len=cfg["data"]["max_time_len"],
        subcarriers=cfg["data"]["subcarriers"],
        use_stft=use_stft,
        nperseg=BEST_STFT[0], noverlap=BEST_STFT[1])

    test_ds = CSIDataset(test_files, class_to_idx,
        min_time_len=cfg["data"]["min_time_len"],
        max_time_len=cfg["data"]["max_time_len"],
        subcarriers=cfg["data"]["subcarriers"],
        use_stft=use_stft,
        nperseg=BEST_STFT[0], noverlap=BEST_STFT[1])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    # 模型（完全读取config，不写死！）
    model = LSTMTransformer(
        input_dim=train_ds[0][0].shape[1],
        hidden_dim=cfg["models"]["lstm_transformer"]["hidden_dim"],
        num_heads=cfg["models"]["lstm_transformer"]["num_heads"],
        num_layers=cfg["models"]["lstm_transformer"]["num_layers"],
        num_classes=7,
        dropout=cfg["models"]["lstm_transformer"]["dropout"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    patience = 0

    # 训练（完全一样）
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for data, lab in pbar:
            data, lab = data.to(DEVICE), lab.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        _, v_acc, _, _ = evaluate_full(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1} | loss={avg_loss:.4f} | val_acc={v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            patience = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ 最优模型已保存")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("🛑 早停")
                break

    # 测试
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    _, t_total_acc, y_true, y_pred = evaluate_full(model, test_loader, DEVICE)
    plot_cm(y_true, y_pred, classes, f"cm_{'stft' if use_stft else 'nostft'}.png")

    # 每个类别准确率
    def get_acc(cls):
        mask = y_true == cls
        if not np.any(mask):
            return 0.0
        return round(accuracy_score(y_true[mask], y_pred[mask]), 4)

    row = {
        "实验": "STFT" if use_stft else "NO_STFT",
        "total_acc": round(t_total_acc, 4),
        "walk": get_acc(0),
        "run": get_acc(1),
        "sitdown": get_acc(2),
        "standup": get_acc(3),
        "fall": get_acc(4),
        "lie_down": get_acc(5),
        "bend": get_acc(6)
    }
    return row

# ===================== 主函数 =====================
def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results = []

    # 实验1：使用STFT
    row1 = run_exp(cfg, use_stft=True)
    results.append(row1)

    # 实验2：不使用STFT
    row2 = run_exp(cfg, use_stft=False)
    results.append(row2)

    # 保存
    save_result_to_excel(results, EXCEL_PATH)

    send_email_with_attachment(
        "✅ STFT消融实验【最终正确版】完成",
        f"STFT: {row1['total_acc']:.2%} | NO_STFT: {row2['total_acc']:.2%}",
        EXCEL_PATH
    )

if __name__ == "__main__":
    main()