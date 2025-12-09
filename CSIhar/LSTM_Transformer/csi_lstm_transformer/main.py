import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import CSIDataset
from src.model import LSTMTransformer
from src.train import train_one_epoch, evaluate
from src.utils import set_seed, collect_files, split_dataset


def main():
    # -----------------------
    # 1. 读取配置
    # -----------------------
    with open("configs/train_config.yaml", 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    os.makedirs(cfg['logging']['out_dir'], exist_ok=True)

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

    # 训练集
    train_ds = CSIDataset(
        train_files, class_to_idx,
        max_time_len=cfg['data']['max_time_len'],
        min_time_len=cfg['data']['min_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=True,
        cache=cfg['data']['cache_in_memory'],
        use_wavelet=cfg['data']['use_wavelet'],
        wavelet_level=cfg['data']['wavelet_level'],
        wavelet_threshold_mode=cfg['data']['wavelet_threshold_mode'],
        use_stft=True
    )

    # 验证集
    val_ds = CSIDataset(
        val_files, class_to_idx,
        max_time_len=cfg['data']['max_time_len'],
        min_time_len=cfg['data']['min_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=False,
        cache=cfg['data']['cache_in_memory'],
        use_wavelet=cfg['data']['use_wavelet'],
        wavelet_level=cfg['data']['wavelet_level'],
        wavelet_threshold_mode=cfg['data']['wavelet_threshold_mode'],
        use_stft=True
    )

    # 测试集（为了训练后测试）
    test_ds = CSIDataset(
        test_files, class_to_idx,
        max_time_len=cfg['data']['max_time_len'],
        min_time_len=cfg['data']['min_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=False,
        cache=cfg['data']['cache_in_memory'],
        use_wavelet=cfg['data']['use_wavelet'],
        wavelet_level=cfg['data']['wavelet_level'],
        wavelet_threshold_mode=cfg['data']['wavelet_threshold_mode'],
        use_stft=True
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    # -----------------------
    # 3. 动态确定 STFT 后的输入维度
    # -----------------------
    sample_input, _ = train_ds[0]
    input_dim = sample_input.shape[1]   # (T, freq_bins) → freq_bins 是输入维度

    # -----------------------
    # 4. 初始化模型
    # -----------------------
    model = LSTMTransformer(
        input_dim=input_dim,
        hidden_dim=cfg['model']['hidden_dim'],
        num_heads=cfg['model']['num_heads'],
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


if __name__ == "__main__":
    main()
