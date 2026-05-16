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
from sklearn.metrics import accuracy_score
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

# ===================== 🔥 双显卡加速 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== STFT参数 =====================
STFT_PARAMS = [
    (2, 1),
]

EXCEL_PATH = "no_stft_ablation_WAVELET.xlsx"  # 没有stft结果表
EPOCHS = 200
PATIENCE = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(EXCEL_PATH):
    os.remove(EXCEL_PATH)

# ===================== 评估函数 =====================
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

def save_result_to_excel(results, excel_path):
    pd.DataFrame(results).to_excel(excel_path, index=False)

def main():
    with open("configs/config_LSTMTransformer.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    classes = cfg["data"]["classes"]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    grouped_files = collect_files(cfg["data"]["raw_root"], classes, cfg["data"]["file_ext"])
    train_files, val_files, test_files = split_dataset(grouped_files, 0.7, 0.15, 0.15)
    results = []

    # ===================== ✅ 固定：有小波变换 =====================
    use_wavelet = True
    wave_label = "有stft"

    for idx, (nseg, novl) in enumerate(STFT_PARAMS, 1):
        print(f"\n==================================================")
        print(f" 第 {idx}/22 组 | STFT=({nseg},{novl}) | {wave_label} | 双显卡")
        print(f"==================================================")

        BEST_MODEL_PATH = f"configs/best_no_stft_{nseg}_{novl}.pth"

        # -----------------------
    # 2. 数据准备
    # -----------------------
        classes = cfg['data']['classes']
        class_to_idx = {c: i for i, c in enumerate(classes)}

        grouped_files = collect_files(
        cfg['data']['raw_root'],
        classes,
        cfg['data']['file_ext']
    )

        train_files, val_files, test_files = split_dataset(
        grouped_files,
        cfg['data']['train_split'],
        cfg['data']['val_split'],
        cfg['data']['test_split'],
        cfg['seed'],
    )

    # ===================== 【数据预处理】 =====================
        nseg = 2
        novl = 1
        train_ds = CSIDataset(train_files, class_to_idx,
                          use_wavelet = True,
            min_time_len=cfg["data"]["min_time_len"],
            max_time_len=cfg["data"]["max_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
           # use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=False,
            nperseg=nseg,
            noverlap=novl
        )

        val_ds = CSIDataset(val_files, class_to_idx,
                         use_wavelet = True,
            min_time_len=cfg["data"]["min_time_len"],
            max_time_len=cfg["data"]["max_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
           # use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=False,
            nperseg=nseg,
            noverlap=novl
        )

        test_ds = CSIDataset(test_files, class_to_idx,
                          use_wavelet = True,
            min_time_len=cfg["data"]["min_time_len"],
            max_time_len=cfg["data"]["max_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
          #  use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=False,
            nperseg=nseg,
            noverlap=novl
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
 

    # -----------------------
    # 3. 动态确定 STFT 后的输入维度
    # -----------------------
        sample_input, _ = train_ds[0]
        input_dim1 = sample_input.shape[1]   # (T, freq_bins) → freq_bins 是输入维度
        model = LSTMTransformer(
            input_dim=input_dim1,
            hidden_dim=cfg['model']['hidden_dim'],
            num_heads=cfg['model']['num_heads'],
            num_layers=cfg['model']['num_layers'],
            num_classes=cfg['model']['num_classes'],
            dropout=cfg['model']['dropout']
        ).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        patience = 0

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
                print(f"✅ 最优模型已保存：{BEST_MODEL_PATH}")
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("🛑 早停")
                    break

        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        _, t_total_acc, y_true, y_pred = evaluate_full(model, test_loader, DEVICE)

        def get_acc(cls):
            mask = y_true == cls
            if not np.any(mask):
                return 0.0
            return round(accuracy_score(y_true[mask], y_pred[mask]), 4)

        row = {
            "nperseg": nseg,
            "noverlap": novl,
            "wavelet": "Yes",
            "total_acc": round(t_total_acc, 4),
            "walk": get_acc(0),
            "run": get_acc(1),
            "sitdown": get_acc(2),
            "standup": get_acc(3),
            "fall": get_acc(4),
            "lie_down": get_acc(5),
            "bend": get_acc(6)
        }
        results.append(row)
        save_result_to_excel(results, EXCEL_PATH)

        send_email_with_attachment(
            f"【无stft有小波】实验{idx}/22完成 | Acc={t_total_acc:.4f}",
            f"STFT({nseg},{novl}) | 无stft有小波变换 | 已完成",
            EXCEL_PATH
        )

    send_email_with_attachment(
        "✅ 全部22组实验完成！【无stft有小波变换】",
        "STFT消融实验（无stft有小波）全部完成",
        EXCEL_PATH
    )

if __name__ == "__main__":
    main()