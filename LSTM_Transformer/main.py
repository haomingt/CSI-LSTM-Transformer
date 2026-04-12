import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

from src.dataset import CSIDataset
from src.model_LSTMTransformer import LSTMTransformer
from src.train import train_one_epoch, evaluate
from src.utils import set_seed, collect_files, split_dataset
from src.model_LSTM import LSTMModel
from src.model_LSTMGMP import LSTM_GMP
from src.utils import set_seed, collect_files, split_dataset
def main():
    # -----------------------
    # 1. 读取配置
    # -----------------------
    a = 1 # 0为LSTM，1加LSTM+Transformer ,2为LSTM+GMP
    text = "configs/config_LSTM.yaml" if a == 0 else "configs/config_LSTMTransformer.yaml"
    if a == 2:
        text = "configs/config_LSTMGMP.yaml"
    with open(text, 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    classes = cfg["data"]["classes"]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    grouped_files = collect_files(cfg["data"]["raw_root"], classes, cfg["data"]["file_ext"])
    train_files, val_files, test_files = split_dataset(grouped_files, 0.7, 0.15, 0.15)

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

        # ===================== 【仅修改：数据预处理 + 训练验证测试逻辑】 =====================
    nseg = 2
    novl = 1
    train_ds = CSIDataset(train_files, class_to_idx,
            min_time_len=cfg["data"]["min_time_len"],
            max_time_len=cfg["data"]["max_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
            use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=True,
            nperseg=nseg,
            noverlap=novl
        )

    val_ds = CSIDataset(val_files, class_to_idx,
            min_time_len=cfg["data"]["min_time_len"],
            max_time_len=cfg["data"]["max_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
            use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=True,
            nperseg=nseg,
            noverlap=novl
        )

    test_ds = CSIDataset(test_files, class_to_idx,
            min_time_len=cfg["data"]["min_time_len"],
            max_time_len=cfg["data"]["max_time_len"],
            subcarriers=cfg["data"]["subcarriers"],
            use_wavelet=cfg["data"]["use_wavelet"],
            wavelet_level=cfg["data"]["wavelet_level"],
            wavelet_threshold_mode=cfg["data"]["wavelet_threshold_mode"],
            use_stft=True,
            nperseg=nseg,
            noverlap=novl
        )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)
 

    # -----------------------
    # 3. 动态确定 STFT 后的输入维度
    # -----------------------
    sample_input, _ = train_ds[0]
    input_dim1 = sample_input.shape[1]   # (T, freq_bins) → freq_bins 是输入维度

    # -----------------------
    # 4. 初始化模型
    # -----------------------
    if a == 0:
        model = LSTMModel(
            input_dim=input_dim1,
            hidden_dim=cfg['model']['hidden_dim'],
            num_layers=cfg['model']['num_layers'],
            num_classes=cfg['model']['num_classes'],
            dropout=cfg['model']['dropout']
        ).to(device)
    elif a == 1:
        model = LSTMTransformer(
            input_dim=input_dim1,
            hidden_dim=cfg['model']['hidden_dim'],
            num_heads=cfg['model']['num_heads'],
            num_layers=cfg['model']['num_layers'],
            num_classes=cfg['model']['num_classes'],
            dropout=cfg['model']['dropout']
        ).to(device)
    else:
        model = LSTM_GMP(
            input_dim=input_dim1,
            hidden_dim=cfg['model']['hidden_dim'],
            num_layers=cfg['model']['num_layers'],
            num_classes=cfg['model']['num_classes'],
            dropout=cfg['model']['dropout']
        ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=float(cfg['training']['weight_decay'])
    )

    scaler = torch.amp.GradScaler('cuda', enabled=cfg['training']['amp'])
    warmup_steps = cfg['training']['warmup_epochs']

    # -----------------------
    # 5. 训练循环（带 early stop）
    # -----------------------
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg['training']['epochs']):
        # ---- 训练 ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer,
            scaler, epoch, warmup_steps
        )

        # ---- 验证 ----
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device, criterion)

        print(f"Epoch {epoch}: 训练损失 {train_loss:.4f}, 训练准确率 {train_acc:.2f}%")
        print(f"           验证损失 {val_loss:.4f}, 验证准确率 {val_acc:.2f}%")

        # ---- 保存最佳模型 ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(cfg['logging']['out_dir'], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型 → 验证准确率 {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= cfg['training']['early_stop_patience']:
                print("早停触发")
                break

    # -----------------------
    # 6. 训练结束 → 加载最佳模型 → 在测试集评估
    # -----------------------
    print("\n========== 加载最佳模型并在测试集评估 ==========")
    model.load_state_dict(torch.load(os.path.join(cfg['logging']['out_dir'], "best_model.pth")))

    test_loss, test_acc, _, _, _ = evaluate(model, test_loader, device, criterion)
    print(f"\n⭐ 最终测试集准确率: {test_acc:.2f}% ⭐")

    # ========================
    # 👇 只在这里加：最优模型计算召回率（完全不改动你之前逻辑）
    # ========================
    print("\n========== 最优模型 - 测试集 召回率 / 精确率 / F1 ==========")
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"✅ 精确率 (Precision): {precision:.4f}")
    print(f"✅ 召回率 (Recall)   : {recall:.4f}")    # 老师要的！
    print(f"✅ F1分数           : {f1:.4f}")

    print("\n===== 每个动作详细指标 =====")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

if __name__ == "__main__":
    main()