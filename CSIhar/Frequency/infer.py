import os
import yaml
import torch
import numpy as np
import json  # 统一导入到顶部
from torch.utils.data import DataLoader, Dataset
# 显式导入所需函数，确保能找到time_to_freq_domain
from csi_dataset import (
    load_csi_file,
    CSIPseudoColorConverter,
    wavelet_denoise_csi,
    time_to_freq_domain  # 直接导入该函数
)
from models.csi_cnn import build_model
from utils.seed import set_seed


class InferenceDataset(Dataset):
    """用于推理的数据集类，继承自基础Dataset"""

    def __init__(self, files, converter, max_time_len, min_time_len, subcarriers,
                 wavelet_level=4, wavelet_threshold_mode='soft'):  # 补充小波参数
        self.files = files
        self.converter = converter
        self.max_time_len = max_time_len
        self.min_time_len = min_time_len
        self.target_time_len = (min_time_len + max_time_len) // 2
        self.subcarriers = subcarriers
        self.freq_bins = 64  # 与训练时保持一致
        self.wavelet_level = wavelet_level
        self.wavelet_threshold_mode = wavelet_threshold_mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        # 加载CSI数据
        csi_time = load_csi_file(fpath)

        # 验证子载波数量
        if csi_time.shape[1] != self.subcarriers:
            raise ValueError(f"子载波数量不匹配: {fpath}, 预期 {self.subcarriers}, 实际 {csi_time.shape[1]}")

        # 使用显式导入的去噪和时频转换函数
        csi_denoised = wavelet_denoise_csi(
            csi_time,
            level=self.wavelet_level,
            threshold_mode=self.wavelet_threshold_mode
        )

        # 转换到频域（直接使用导入的time_to_freq_domain）
        csi_freq = time_to_freq_domain(csi_denoised, freq_bins=self.freq_bins)

        # 转换为伪彩色图像
        img = self.converter(csi_freq)
        img = np.ascontiguousarray(img, dtype=np.float32)

        return torch.from_numpy(img), fpath


def main():
    import os
    import yaml
    import torch
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    from torch.utils.data import DataLoader, Dataset
    from csi_dataset import (
        load_csi_file,
        CSIPseudoColorConverter,
        wavelet_denoise_csi,
        time_to_freq_domain
    )
    from models.csi_cnn import build_model
    from utils.seed import set_seed

    class InferenceDataset(Dataset):
        """用于推理的数据集类，继承自基础Dataset"""

        def __init__(self, files, converter, max_time_len, min_time_len, subcarriers,
                     wavelet_level=4, wavelet_threshold_mode='soft', with_label=True):
            self.files = files
            self.converter = converter
            self.max_time_len = max_time_len
            self.min_time_len = min_time_len
            self.target_time_len = (min_time_len + max_time_len) // 2
            self.subcarriers = subcarriers
            self.freq_bins = 64
            self.wavelet_level = wavelet_level
            self.wavelet_threshold_mode = wavelet_threshold_mode
            self.with_label = with_label  # 是否包含标签信息

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            fpath = self.files[idx]
            # 加载CSI数据
            csi_time = load_csi_file(fpath)

            # 验证子载波数量
            if csi_time.shape[1] != self.subcarriers:
                raise ValueError(f"子载波数量不匹配: {fpath}, 预期 {self.subcarriers}, 实际 {csi_time.shape[1]}")

            # 去噪和时频转换
            csi_denoised = wavelet_denoise_csi(
                csi_time,
                level=self.wavelet_level,
                threshold_mode=self.wavelet_threshold_mode
            )

            csi_freq = time_to_freq_domain(csi_denoised, freq_bins=self.freq_bins)

            # 转换为伪彩色图像
            img = self.converter(csi_freq)
            img = np.ascontiguousarray(img, dtype=np.float32)

            if self.with_label:
                # 从文件路径提取标签（假设路径中包含类别文件夹）
                label = os.path.basename(os.path.dirname(fpath))
                return torch.from_numpy(img), fpath, label
            return torch.from_numpy(img), fpath

    def compute_metrics(labels, preds, classes, out_dir):
        """计算评估指标并保存混淆矩阵和报告"""
        # 确保输出目录存在
        os.makedirs(out_dir, exist_ok=True)

        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(out_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

        # 生成分类报告
        report = classification_report(labels, preds, target_names=classes)

        # 保存分类报告到TXT文件
        report_path = os.path.join(out_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return cm, report

    def main():
        # 配置参数
        config_path = "./train_config.yaml"
        model_path = "./best_model/best_model.pth"
        data_dir = "../data_raw_2"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 填充默认配置
        from train import fill_defaults
        fill_defaults(cfg)

        # 设置随机种子
        set_seed(cfg['seed'])

        # 收集待推理的文件
        file_ext = cfg['data']['file_ext']
        infer_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(file_ext):
                    infer_files.append(os.path.join(root, file))

        if not infer_files:
            print(f"未找到任何{file_ext}文件在 {data_dir} 目录下")
            return

        print(f"找到 {len(infer_files)} 个待推理文件")

        # 初始化数据转换器
        converter = CSIPseudoColorConverter(
            target_size=cfg['data']['resize'],
            cmap=cfg['data']['cmap'],
            norm_mode=cfg['data']['norm_mode']
        )

        # 全局归一化参数
        if cfg['data']['norm_mode'] == 'global':
            print("计算全局归一化参数...")
            mats = [load_csi_file(fp) for fp in infer_files[:2000]]
            converter.fit_global(mats)

        # 创建推理数据集（包含标签提取）
        classes = cfg['data']['classes']
        class_to_idx = {c: i for i, c in enumerate(classes)}

        infer_dataset = InferenceDataset(
            infer_files,
            converter,
            max_time_len=cfg['data']['max_time_len'],
            min_time_len=cfg['data']['min_time_len'],
            subcarriers=cfg['data']['subcarriers'],
            wavelet_level=4,
            wavelet_threshold_mode='soft',
            with_label=True  # 启用标签提取
        )

        infer_loader = DataLoader(
            infer_dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True
        )

        # 构建模型
        model = build_model(
            cfg['model']['name'],
            num_classes=len(classes),
            in_channels=cfg['model']['in_channels'],
            dropout=cfg['model']['dropout']
        ).to(device)

        # 加载模型权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # 推理过程
        results = []
        all_preds = []
        all_labels = []
        print("开始推理...")
        with torch.no_grad():
            for imgs, fpaths, labels in infer_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                # 收集预测结果和标签用于计算指标
                pred_classes = [classes[p] for p in preds.cpu().numpy()]
                all_preds.extend(pred_classes)
                all_labels.extend(labels)

                # 保存详细结果
                for i, fpath in enumerate(fpaths):
                    pred_class = classes[preds[i]]
                    pred_prob = probabilities[i][preds[i]].item()
                    results.append({
                        'file_path': fpath,
                        'true_class': labels[i],
                        'predicted_class': pred_class,
                        'confidence': pred_prob,
                        'all_probabilities': {classes[j]: probabilities[i][j].item() for j in range(len(classes))}
                    })

        # 输出详细结果
        for res in results:
            print(f"文件: {res['file_path']}")
            print(f"真实类别: {res['true_class']}")
            print(f"预测类别: {res['predicted_class']} (置信度: {res['confidence']:.4f})")
            print("类别概率:")
            for cls, prob in res['all_probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
            print("---")

        # 创建结果输出目录（与infer1.py保持一致）
        model_out_dir = os.path.join(os.path.dirname(model_path), "test_results")
        os.makedirs(model_out_dir, exist_ok=True)

        # 计算并保存评估指标（混淆矩阵和分类报告）
        cm, report = compute_metrics(all_labels, all_preds, classes, model_out_dir)
        print("分类报告:\n", report)

        # 保存详细推理结果到JSON
        json_path = os.path.join(model_out_dir, "inference_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存简单结果到TXT
        txt_path = os.path.join(model_out_dir, "inference_summary.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("推理结果汇总\n")
            f.write("=" * 50 + "\n")
            for res in results:
                f.write(f"文件: {res['file_path']}\n")
                f.write(f"真实类别: {res['true_class']}\n")
                f.write(f"预测类别: {res['predicted_class']} (置信度: {res['confidence']:.4f})\n")
                f.write("-" * 50 + "\n")

        print(f"推理结果已保存到 {model_out_dir} 目录")
        print(f"混淆矩阵已保存到 {os.path.join(model_out_dir, 'confusion_matrix.png')}")
        print(f"分类报告已保存到 {os.path.join(model_out_dir, 'classification_report.txt')}")

    if __name__ == "__main__":
        main()

if __name__ == "__main__":
    main()