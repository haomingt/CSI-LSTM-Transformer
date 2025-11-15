import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def compute_metrics(y_true, y_pred, class_names, out_dir=None):
    """
    计算并可视化混淆矩阵 + 分类报告。
    y_true / y_pred: List[int]
    class_names: 类别名称列表
    out_dir: 若提供，则保存图像与报告文本
    """
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # 混淆矩阵图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        # 分类报告文本（保持原有逻辑）
        with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        # 新增：Excel输出（修改部分）
        report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)

        # 提取数据
        rows = []
        for cls in class_names:
            metrics = report_dict[cls]
            rows.append({
                "类别": cls,
                "精确率(precision)": f"{metrics['precision']:.4f}",
                "召回率(recall)": f"{metrics['recall']:.4f}",
                "F1分数(f1-score)": f"{metrics['f1-score']:.4f}",
                "支持样本数(support)": metrics['support']
            })

        # 计算accuracy
        total_preds = len(y_pred)
        correct_preds = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        rows.append({
            "类别": "accuracy",
            "精确率(precision)": "-",
            "召回率(recall)": "-",
            "F1分数(f1-score)": f"{accuracy:.4f}",
            "支持样本数(support)": total_preds
        })

        # 处理汇总行
        for key in ["macro avg", "weighted avg"]:
            if key in report_dict:
                metrics = report_dict[key]
                rows.append({
                    "类别": key,
                    "精确率(precision)": f"{metrics['precision']:.4f}",
                    "召回率(recall)": f"{metrics['recall']:.4f}",
                    "F1分数(f1-score)": f"{metrics['f1-score']:.4f}",
                    "支持样本数(support)": metrics["support"]
                })

        # 保存为Excel
        df = pd.DataFrame(rows)
        df.to_excel(os.path.join(out_dir, "classification_report.xlsx"), index=False)

    return cm, report


def accuracy_topk(output, target, topk=(1,)):
    """
    计算 Top-K 准确率
    output: (B, C) logits
    target: (B,)
    返回: List[float] 与 topk 对应的百分比准确率
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # 取出每行topk的类别索引
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, B)
    # 将target展开并与pred对比
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res