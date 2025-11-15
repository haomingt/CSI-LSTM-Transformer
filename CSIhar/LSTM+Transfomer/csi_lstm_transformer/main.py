import os
import yaml
import torch
from src.dataset import CSIDataset
from src.model import LSTMTransformer
from src.train import train_one_epoch, evaluate
from src.utils import set_seed, collect_files, split_dataset
from torch.utils.data import DataLoader


def main():
    # 加载配置文件
    with open("configs/train_config.yaml", 'r',encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 初始化设置
    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(cfg['logging']['out_dir'], exist_ok=True)

    # 数据准备部分修改
    classes = cfg['data']['classes']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # 收集分组文件（按类别-用户）
    grouped_files = collect_files(cfg['data']['raw_root'], classes, cfg['data']['file_ext'])
    total_files = 0
    for cls_dict in grouped_files.values():  # 每个类别对应的 {用户: [文件列表]} 字典
        for file_list in cls_dict.values():  # 每个用户对应的文件列表
            total_files += len(file_list)
    print(f"找到 {total_files} 个数据文件")

    # 划分数据集（按用户均衡划分）
    train_files, val_files, test_files = split_dataset(
        grouped_files,
        cfg['data']['train_split'],
        cfg['data']['val_split'],
        cfg['data']['test_split'],
        cfg['seed']
    )
    print(f"训练集: {len(train_files)} 个样本, 验证集: {len(val_files)} 个样本, 测试集: {len(test_files)} 个样本")

    # 创建数据集和数据加载器（传递小波参数）
    train_ds = CSIDataset(
        train_files, class_to_idx,
        max_time_len=cfg['data']['max_time_len'],
        min_time_len=cfg['data']['min_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=True,
        cache=cfg['data']['cache_in_memory'],
        use_wavelet=cfg['data']['use_wavelet'],
        wavelet_level=cfg['data']['wavelet_level'],
        wavelet_threshold_mode=cfg['data']['wavelet_threshold_mode']
    )
    val_ds = CSIDataset(
        val_files, class_to_idx,
        max_time_len=cfg['data']['max_time_len'],
        min_time_len=cfg['data']['min_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=False,
        cache=cfg['data']['cache_in_memory'],
        use_wavelet=cfg['data']['use_wavelet'],
        wavelet_level=cfg['data']['wavelet_level'],
        wavelet_threshold_mode=cfg['data']['wavelet_threshold_mode']
    )

    # 测试集同样修改
    test_ds = CSIDataset(
        test_files, class_to_idx,
        max_time_len=cfg['data']['max_time_len'],
        min_time_len=cfg['data']['min_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=False,
        cache=cfg['data']['cache_in_memory'],
        use_wavelet=cfg['data']['use_wavelet'],
        wavelet_level=cfg['data']['wavelet_level'],
        wavelet_threshold_mode=cfg['data']['wavelet_threshold_mode']
    )

    # 在创建数据集后添加数据加载器
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LSTMTransformer(
        input_dim=cfg['data']['subcarriers'] * 2,  # <- 注意这里
        hidden_dim=cfg['model']['hidden_dim'],
        num_heads=cfg['model']['num_heads'],
        num_layers=cfg['model']['num_layers'],
        num_classes=cfg['model']['num_classes'],
        dropout=cfg['model']['dropout']
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # 找到优化器定义部分，修改为：
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=float(cfg['training']['weight_decay'])  # 强制转换为浮点数
    )

    scaler = torch.amp.GradScaler('cuda', enabled=cfg['training']['amp'])

    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    warmup_steps = cfg['training']['warmup_epochs']

    for epoch in range(cfg['training']['epochs']):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer,
            scaler, epoch, warmup_steps
        )

        # 验证
        val_loss, val_acc, _, _, _ = evaluate(  # 增加一个下划线接收报告返回值
            model, val_loader, device, criterion
        )

        print(f"Epoch {epoch}: 训练损失 {train_loss:.4f}, 训练准确率 {train_acc:.2f}%")
        print(f"         验证损失 {val_loss:.4f}, 验证准确率 {val_acc:.2f}%")

        # 早停和保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg['logging']['out_dir'], "best_model.pth"))
            print(f"保存新的最佳模型，验证准确率: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= cfg['training']['early_stop_patience']:
                print(f"早停触发，最佳验证准确率: {best_val_acc:.2f}%")
                break

    # 测试最终模型
    test_ds = CSIDataset(
        test_files, class_to_idx,
        max_time_len=cfg['data']['max_time_len'],
        min_time_len=cfg['data']['min_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=False,
        cache=cfg['data']['cache_in_memory']
    )
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    # 定义报告保存路径（outputs/test_results/）
    test_result_path = os.path.join(cfg['logging']['out_dir'], "test_results")
    model.load_state_dict(torch.load(os.path.join(cfg['logging']['out_dir'], "best_model.pth")))

    # 调用evaluate时传入classes和save_path
    test_loss, test_acc, test_preds, test_labels, test_report = evaluate(
        model, test_loader, device, criterion,
        classes=classes,  # 传入动作类别列表
        save_path=test_result_path  # 传入报告保存路径
    )
    print(f"\n测试结果: 损失 {test_loss:.4f}, 准确率 {test_acc:.2f}%")
    print("\n分类报告:")
    print(test_report)  # 打印报告到控制台


if __name__ == "__main__":
    main()
