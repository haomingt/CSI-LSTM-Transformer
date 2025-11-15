"""
多模型推理评估：修复CSIDataset初始化参数错误 + 确保生成结果文件夹
"""
import argparse
import yaml
import torch
import numpy as np
import os  # 确保导入os模块
import sys
from torch.utils.data import DataLoader

# 获取当前脚本所在目录并添加项目根目录到搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.transforms import CSIPseudoColorConverter
from datasets.csi_dataset import CSIDataset
from models.csi_cnn import build_model
from utils.metrics import compute_metrics
from train import evaluate  # 从train.py导入评估函数


# 收集测试集文件的函数
def collect_test_files(testset_path, file_ext):
    test_files = []
    for root, dirs, files in os.walk(testset_path):
        for file in files:
            if file.lower().endswith(file_ext.lower()):
                file_path = os.path.join(root, file)
                file_path = os.path.normpath(file_path)
                test_files.append(file_path)
    return test_files


def load_model(cfg, model_path, device):
    num_classes = len(cfg['data']['classes'])
    model = build_model(
        cfg['model']['name'],
        num_classes=num_classes,
        in_channels=cfg['model']['in_channels'],
        dropout=cfg['model']['dropout']
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--model1", required=True, help="第一个模型权重文件路径")
    parser.add_argument("--model2", required=True, help="第二个模型权重文件路径")
    parser.add_argument("--testset", required=True, help="测试集文件夹路径")
    parser.add_argument("--device", default="cuda", help="计算设备（cuda或cpu）")
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    classes = cfg['data']['classes']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 收集测试集文件
    file_ext = cfg['data']['file_ext']
    test_files = collect_test_files(args.testset, file_ext)

    if not test_files:
        print(f"错误：在 {args.testset} 中未找到 {file_ext} 格式的文件")
        print("\n文件夹中的文件列表：")
        for root, dirs, files in os.walk(args.testset):
            for file in files:
                print(f"  - {os.path.join(root, file)}")
        return
    print(f"已加载测试集，共 {len(test_files)} 个 {file_ext} 文件")

    # 创建数据转换器
    converter = CSIPseudoColorConverter(
        target_size=cfg['data']['resize'],
        cmap=cfg['data']['cmap'],
        norm_mode=cfg['data']['norm_mode']
    )

    # 处理全局归一化
    if cfg['data']['norm_mode'] == 'global':
        from datasets.csi_dataset import load_csi_file, collect_files as original_collect
        print("[INFO] 应用全局归一化参数...")
        train_files = original_collect(
            root=cfg['data']['raw_root'],
            classes=classes,
            file_ext=file_ext
        )
        mats = [load_csi_file(fp) for i, fp in enumerate(train_files) if i <= 2000]
        converter.fit_global(mats)

    # 创建测试数据集（关键修改：根据CSIDataset实际参数调整）
    test_dataset = CSIDataset(
        test_files,  # 直接传入文件列表（不使用data_list关键字）
        class_to_idx,
        converter,
        max_time_len=cfg['data']['max_time_len'],
        subcarriers=cfg['data']['subcarriers'],
        augment=False,
        cache=cfg['data']['cache_in_memory'],
        min_time_len=cfg['data']['min_time_len']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # -------------------------- 修复1：评估模型1前创建结果文件夹 --------------------------
    print(f"\n[INFO] 评估第一个模型: {args.model1}")
    model1 = load_model(cfg, args.model1, device)
    test_loss1, test_acc1, test_topk1, test_preds1, test_labels1 = evaluate(
        model1, test_loader, device, criterion, cfg
    )
    print(f"[模型1测试结果] Loss={test_loss1:.4f} Acc={test_acc1:.2f}% TopK={test_topk1}")

    # 1. 定义模型1的结果文件夹路径
    model1_out_dir = os.path.join(os.path.dirname(args.model1), "test_results_model1")
    # 2. 手动创建文件夹（exist_ok=True：若已存在则不报错）
    os.makedirs(model1_out_dir, exist_ok=True)
    # 3. 调用compute_metrics（此时文件夹已存在，可正常保存结果）
    cm1, report1 = compute_metrics(
        test_labels1,
        test_preds1,
        classes,
        out_dir=model1_out_dir  # 传入已创建的文件夹路径
    )
    print("模型1分类报告:\n", report1)

    # -------------------------- 修复2：评估模型2前创建结果文件夹 --------------------------
    print(f"\n[INFO] 评估第二个模型: {args.model2}")
    model2 = load_model(cfg, args.model2, device)
    test_loss2, test_acc2, test_topk2, test_preds2, test_labels2 = evaluate(
        model2, test_loader, device, criterion, cfg
    )
    print(f"[模型2测试结果] Loss={test_loss2:.4f} Acc={test_acc2:.2f}% TopK={test_topk2}")

    # 1. 定义模型2的结果文件夹路径
    model2_out_dir = os.path.join(os.path.dirname(args.model2), "test_results_model2")
    # 2. 手动创建文件夹
    os.makedirs(model2_out_dir, exist_ok=True)
    # 3. 调用compute_metrics
    cm2, report2 = compute_metrics(
        test_labels2,
        test_preds2,
        classes,
        out_dir=model2_out_dir  # 传入已创建的文件夹路径
    )
    print("模型2分类报告:\n", report2)

    # 对比结果
    print("\n[模型对比]")
    print(f"准确率差异 (模型1 - 模型2): {test_acc1 - test_acc2:.2f}%")
    print(f"TopK准确率差异 (模型1 - 模型2): {[round(a-b, 2) for a, b in zip(test_topk1, test_topk2)]}%")
    print(f"损失差异 (模型1 - 模型2): {test_loss1 - test_loss2:.4f}")


if __name__ == "__main__":
    main()