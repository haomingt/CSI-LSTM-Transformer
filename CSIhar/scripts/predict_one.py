# scripts/predict_one.py
import yaml, torch
from models.csi_cnn import build_model
from utils.transforms import CSIPseudoColorConverter
from datasets.csi_dataset import load_csi_file, pad_or_truncate
import sys

def load_everything(config_path="outputs/used_config.yaml",
                    weights_path="outputs/best_model.pth",
                    device=None):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    classes = cfg['data']['classes']
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建模型
    model = build_model(cfg['model']['name'],
                        num_classes=len(classes),
                        in_channels=cfg['model']['in_channels'],
                        dropout=cfg['model']['dropout'])
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state['model'])
    model.to(device).eval()

    # 构建转换器
    converter = CSIPseudoColorConverter(
        target_size=cfg['data']['resize'],
        cmap=cfg['data']['cmap'],
        norm_mode=cfg['data']['norm_mode']
    )

    # 如果是 global 归一化，需确保训练阶段已把 global_min/global_max 保存进 cfg
    if cfg['data']['norm_mode'] == 'global':
        gmin = cfg['data'].get('global_min')
        gmax = cfg['data'].get('global_max')
        if gmin is None or gmax is None:
            raise ValueError("需要 global_min/global_max（训练时保存），否则无法保持一致。")
        converter.global_min = float(gmin)
        converter.global_max = float(gmax)

    return cfg, classes, model, converter, device

def predict_one(file_path, cfg, classes, model, converter, device):
    mat = load_csi_file(file_path)              # (T,F)
    if mat.shape[1] != cfg['data']['subcarriers']:
        raise ValueError("子载波数不匹配")
    mat = pad_or_truncate(mat, cfg['data']['max_time_len'])
    img = converter(mat)                        # (C,H,W)
    x = torch.from_numpy(img).unsqueeze(0).to(device)   # (1,C,H,W)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top1_idx = int(probs.argmax().item())
    return classes[top1_idx], float(probs[top1_idx].item())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m scripts.predict_one path/to/sample.csv")
        sys.exit(0)
    file_path = sys.argv[1]
    cfg, classes, model, converter, device = load_everything()
    cls, p = predict_one(file_path, cfg, classes, model, converter, device)
    print(f"预测结果: {cls} (概率 {p:.4f})")
