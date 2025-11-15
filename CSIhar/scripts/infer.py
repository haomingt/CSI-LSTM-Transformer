"""
单样本推理：
  输入一个原始 CSI 文件 (csv/npy)，输出 top-3 类别概率。
"""
import argparse
import yaml
import torch
import numpy as np

from utils.transforms import CSIPseudoColorConverter
from datasets.csi_dataset import load_csi_file, pad_or_truncate
from models.csi_cnn import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/train_config.yaml", help="配置文件")
    parser.add_argument("--model", default="./outputs/best_model.pth", help="已训练模型权重")
    parser.add_argument("--input", required=True, help="待预测 CSI 文件路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=3, help="显示TopK结果")
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    classes = cfg['data']['classes']
    num_classes = len(classes)

    # 构建模型
    model = build_model(cfg['model']['name'],
                        num_classes=num_classes,
                        in_channels=cfg['model']['in_channels'],
                        dropout=cfg['model']['dropout'])

    ckpt = torch.load(args.model, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 伪彩色转换器 (推理时一般使用 per_sample 即可)
    converter = CSIPseudoColorConverter(
        target_size=cfg['data']['resize'],
        cmap=cfg['data']['cmap'],
        norm_mode='per_sample'
    )

    # 加载 CSI
    mat = load_csi_file(args.input)
    mat = pad_or_truncate(mat, cfg['data']['max_time_len'])

    img = converter(mat)          # (C,H,W)
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,C,H,W)

    with torch.no_grad():
        logits = model(img_t)
        prob = torch.softmax(logits, dim=1)
        top_vals, top_idxs = torch.topk(prob, k=min(args.topk, num_classes), dim=1)

    print(f"[RESULT] 文件: {args.input}")
    for rank, (score, idx) in enumerate(zip(top_vals[0], top_idxs[0]), start=1):
        print(f"Top{rank}: {classes[idx.item()]}  概率={score.item():.4f}")


if __name__ == "__main__":
    main()
