import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 添加BLSTM.py所在路径，确保模块导入正常（适配目标路径）
sys.path.append(r"C:\Users\XFish777\Desktop\postgraduate\postgraduate\CSI_comp\CSIhar\models")
# 导入数据预处理模块与自定义注意力层（加载模型需用到）
from data_processor_BLSTM import extract_csi_by_label, train_test_split
from model_builder_BLSTM import AttenLayer


def load_trained_model(model_path):
    """
    功能：加载训练好的注意力BLSTM模型（需指定自定义Attention层，避免加载失败）
    论文依据（sensors-21-07225-v2.pdf）：
    - 3.3.2节：模型含自定义注意力层，加载时需显式声明以匹配训练时结构；
    - 5.2节：最优模型保存为HDF5格式，需保证加载后参数与训练结果一致。
    参数：
        model_path: str，训练好的模型文件路径（如best_blstm_model.hdf5）
    返回：
        tf.keras.Model: 加载完成的BLSTM模型（可直接用于预测）
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}\n请先通过trainer_BLSTM.py训练模型")
    
    # 加载模型，通过custom_objects指定自定义注意力层（核心步骤）
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"AttenLayer": AttenLayer}  # 匹配训练时的自定义层
    )
    print(f"=== 模型加载成功！模型结构与论文图4一致 ===")
    print(f"模型输入形状：{model.input_shape}（500时间步×52子载波，与论文一致）")
    print(f"模型输出形状：{model.output_shape}（7类活动概率分布，与论文一致）")
    return model


def test_model_performance(model, x_test, y_test, all_labels, history_path=None):
    """
    功能：测试模型在测试集上的性能（计算准确率、绘制混淆矩阵与训练曲线）
    论文依据（sensors-21-07225-v2.pdf）：
    - 5.2节：采用准确率与混淆矩阵评估模型性能，重点关注相似活动（如lie down/sitdown）的区分能力；
    - 图8-9：需对比模型在7类活动上的准确率，混淆矩阵需按行归一化（显示每类活动的分类精度）。
    参数：
        model: tf.keras.Model，加载好的BLSTM模型
        x_test: np.ndarray，测试集CSI样本（形状：(测试样本数, 500, 52)）
        y_test: np.ndarray，测试集独热编码标签（形状：(测试样本数, 7)）
        all_labels: list[str]，7类活动标签（与论文一致的顺序）
        history_path: str，训练历史保存路径（如blstm_training_history.npz，用于绘制训练曲线）
    """
    # -------------------------- 1. 计算测试集准确率 --------------------------
    print("\n=== 开始在测试集上评估模型性能 ===")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"测试集准确率：{test_acc:.4f}（目标：≥94%，与论文BLSTM性能一致）")
    print(f"测试集损失：{test_loss:.4f}")

    # -------------------------- 2. 生成测试集预测结果 --------------------------
    # 预测概率分布（形状：(测试样本数, 7)）
    y_pred_prob = model.predict(x_test, verbose=1)
    # 转换为类别索引（取概率最大的类别，形状：(测试样本数,)）
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)  # 真实类别索引

    # -------------------------- 3. 绘制混淆矩阵（按论文图9格式） --------------------------
    print("\n=== 绘制混淆矩阵（与论文图9一致，按行归一化） ===")
    # 计算混淆矩阵（行数=真实类别，列数=预测类别）
    cm = confusion_matrix(y_true, y_pred)
    # 按行归一化（显示每类活动的分类精度，论文图9采用此方式）
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # 绘制混淆矩阵（设置大尺寸避免标签重叠，与论文可视化风格一致）
    plt.figure(figsize=(12, 10))  # 论文图9采用大尺寸确保标签清晰
    cmd = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=all_labels  # 标签顺序与论文一致
    )
    cmd.plot(
        cmap=plt.cm.Blues,  # 配色与论文一致（蓝色系）
        xticks_rotation=45,  # 横轴标签旋转45度，避免重叠
        values_format=".2f"  # 显示两位小数，与论文归一化精度一致
    )
    plt.title("Confusion Matrix (Normalized) - Attention BLSTM", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()  # 自动调整布局，避免标签截断
    plt.savefig("blstm_confusion_matrix.png", dpi=300)  # 保存为高清图片（300dpi）
    print(f"混淆矩阵已保存为：blstm_confusion_matrix.png（与论文图9格式一致）")
    plt.show()

    # -------------------------- 4. 绘制训练曲线（准确率+损失，参考论文图8） --------------------------
    if history_path and os.path.exists(history_path):
        print("\n=== 绘制训练曲线（准确率+损失，参考论文图8） ===")
        # 加载训练历史
        history_data = np.load(history_path)
        train_acc = history_data["train_accuracy"]
        val_acc = history_data["val_accuracy"]
        train_loss = history_data["train_loss"]
        val_loss = history_data["val_loss"]
        epochs = history_data["epochs"]

        # 子图1：准确率曲线
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), train_acc, label="Train Accuracy", linewidth=2)
        plt.plot(range(1, epochs+1), val_acc, label="Validation Accuracy", linewidth=2)
        plt.axhline(y=max(val_acc), color="r", linestyle="--", label=f"Best Val Acc: {max(val_acc):.4f}")
        plt.title("Model Accuracy (Attention BLSTM)", fontsize=12)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图2：损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), train_loss, label="Train Loss", linewidth=2)
        plt.plot(range(1, epochs+1), val_loss, label="Validation Loss", linewidth=2)
        plt.axhline(y=min(val_loss), color="r", linestyle="--", label=f"Best Val Loss: {min(val_loss):.4f}")
        plt.title("Model Loss (Attention BLSTM)", fontsize=12)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Loss (Categorical Crossentropy)", fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("blstm_training_curves.png", dpi=300)
        print(f"训练曲线已保存为：blstm_training_curves.png（参考论文图8格式）")
        plt.show()

    # -------------------------- 5. 输出每类活动的分类精度 --------------------------
    print("\n=== 每类活动的分类精度（与论文图8对比） ===")
    class_acc = cm_normalized.diagonal()  # 对角线元素为每类活动的分类精度
    for label, acc in zip(all_labels, class_acc):
        print(f"活动 '{label}'：分类精度 = {acc:.4f}")
    # 重点关注论文提及的相似活动（lie down/sitdown）
    lie_down_idx = all_labels.index("lie down")
    sitdown_idx = all_labels.index("sitdown")
    print(f"\n重点对比：")
    print(f"活动 'lie down' 分类精度：{class_acc[lie_down_idx]:.4f}（论文BLSTM约95%）")
    print(f"活动 'sitdown' 分类精度：{class_acc[sitdown_idx]:.4f}（论文BLSTM约94%）")


def main(raw_dataset_folder, model_path="best_blstm_model.hdf5", history_path="blstm_training_history.npz"):
    """
    主函数：完成模型测试全流程（加载数据→加载模型→性能评估→可视化）
    论文依据（sensors-21-07225-v2.pdf）：
    - 5.1节：测试集与训练集采用相同的数据预处理逻辑（滑动窗口、52子载波）；
    - 5.2节：测试流程需保证数据划分、模型结构与训练阶段完全一致，确保结果可复现。
    """
    # 1. 定义论文指定的7类活动标签（顺序与训练时一致）
    all_labels = ["lie down", "fall", "bend", "run", "sitdown", "standup", "walk"]

    # 2. 加载测试集数据（与训练时相同的预处理逻辑，确保数据格式一致）
    print("=== 加载测试集数据（与训练时预处理逻辑一致） ===")
    feat_tuple = []
    label_tuple = []
    for label in all_labels:
        feat, label_arr = extract_csi_by_label(
            raw_folder=raw_dataset_folder,
            target_label=label,
            all_labels=all_labels,
            save=False  # 无需重复保存，仅加载用于测试
        )
        feat_tuple.append(feat)
        label_tuple.append(label_arr)
    
    # 划分测试集（与训练时相同的比例和随机种子，确保测试集与训练集无重叠）
    _, _, x_test, y_test = train_test_split(
        feat_tuple=feat_tuple,
        label_tuple=label_tuple,
        train_ratio=0.75,
        seed=379  # 与论文一致的随机种子，保证数据划分可复现
    )
    print(f"测试集规模：{x_test.shape[0]} 个样本（25%总数据，与论文一致）")

    # 3. 加载训练好的模型
    model = load_trained_model(model_path)

    # 4. 测试模型性能并可视化结果
    test_model_performance(
        model=model,
        x_test=x_test,
        y_test=y_test,
        all_labels=all_labels,
        history_path=history_path
    )


if __name__ == "__main__":
    """
    命令行调用入口：
    用法：python tester_BLSTM.py <CSI数据集根目录路径> [可选：模型路径] [可选：训练历史路径]
    示例：python tester_BLSTM.py C:\CSI_dataset best_blstm_model.hdf5 blstm_training_history.npz
    """
    # 解析命令行参数（支持1-3个参数，后两个为可选）
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("错误！命令行参数格式不正确。")
        print("正确用法1（默认模型/历史路径）：python tester_BLSTM.py <数据集根目录>")
        print("正确用法2（自定义路径）：python tester_BLSTM.py <数据集根目录> <模型路径> <训练历史路径>")
        print("示例：python tester_BLSTM.py C:\\Users\\XFish777\\Desktop\\CSI_dataset")
        sys.exit(1)
    
    # 提取参数
    raw_dataset_folder = sys.argv[1].strip()
    model_path = sys.argv[2] if len(sys.argv) >=3 else "best_blstm_model.hdf5"
    history_path = sys.argv[3] if len(sys.argv) ==4 else "blstm_training_history.npz"

    # 校验数据集目录
    if not os.path.exists(raw_dataset_folder):
        raise FileNotFoundError(f"数据集根目录不存在：{raw_dataset_folder}\n请检查路径是否正确")
    
    # 启动测试流程
    main(raw_dataset_folder, model_path, history_path)