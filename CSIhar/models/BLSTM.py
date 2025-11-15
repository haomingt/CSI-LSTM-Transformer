import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import os
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm


# 数据处理部分保持不变（与TensorFlow版本共用）
def merge_csi_label(csifile, labelfile, win_len=500, thrshd=0.6, step=50):
    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label = line[0]
            activity.append(0 if label == 'NoActivity' else 1)
    activity = np.array(activity)

    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])[:52]  # 仅保留幅度特征
            csi.append(line_array[np.newaxis, ...])
    csi = np.concatenate(csi, axis=0)
    assert csi.shape[0] == activity.shape[0]

    feature = []
    index = 0
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index + win_len]
        if np.sum(cur_activity) < thrshd * win_len:
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 52))
        cur_feature[0] = csi[index:index + win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)


def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=500, thrshd=0.6, step=50):
    print(f'Starting Extract CSI for Label {label}')
    label = label.lower()
    if label not in labels:
        raise ValueError(f"The label {label} should be among {labels}")

    data_path_pattern = os.path.join(raw_folder, label, f'user_*{label}*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    annot_csv_files = sorted(glob.glob(os.path.join(raw_folder, label, f'Annotation_user_*{label}*.csv')))

    feature = []
    for i, (csi_file, label_file) in enumerate(zip(input_csv_files, annot_csv_files)):
        if not os.path.exists(label_file):
            print(f'Warning! Label File {label_file} doesn\'t exist.')
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len, thrshd, step))
        print(f'Finished {((i + 1) / len(input_csv_files)) * 100:.2f}% for Label {label}')

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed(f"X_{label}_win_{win_len}_thrshd_{int(thrshd * 100)}percent_step_{step}.npz", feat_arr)

    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def extract_csi(raw_folder, labels, save=False, win_len=500, thrshd=0.6, step=50):
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


# PyTorch数据集类
class CSIDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.FloatTensor(x_data)
        self.y = torch.FloatTensor(y_data)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 注意力层实现
class AttenLayer(nn.Module):
    def __init__(self, num_state):
        super(AttenLayer, self).__init__()
        self.num_state = num_state

    def build(self, input_shape):
        self.kernel = nn.Parameter(torch.randn(input_shape[-1], self.num_state))
        self.bias = nn.Parameter(torch.randn(self.num_state))
        self.prob_kernel = nn.Parameter(torch.randn(self.num_state))

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape

        # 初始化参数（首次调用时）
        if not hasattr(self, 'kernel'):
            self.build(x.shape)

        # 计算注意力权重
        atten_state = torch.tanh(torch.matmul(x, self.kernel) + self.bias)  # (batch, seq_len, num_state)
        logits = torch.matmul(atten_state, self.prob_kernel)  # (batch, seq_len)
        prob = torch.softmax(logits, dim=1)  # (batch, seq_len)

        # 加权求和
        weighted_feature = torch.sum(x * prob.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return weighted_feature


# 模型配置类
class CSIModelConfig:
    def __init__(self, win_len=500, step=50, thrshd=0.6, downsample=1):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("lie down", "fall", "bend", "run", "sitdown", "standup", "walk")
        self._downsample = downsample
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocessing(self, raw_folder, save=False):
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        if self._downsample > 1:
            return tuple([v[:, ::self._downsample, ...] if i % 2 == 0 else v
                          for i, v in enumerate(numpy_tuple)])
        return numpy_tuple

    def load_csi_data_from_files(self, np_files):
        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for the 7 activities')
        x = [np.load(f)['arr_0'] for f in np_files]
        if self._downsample > 1:
            x = [arr[:, ::self._downsample, :] for arr in x]

        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:, i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)

    def build_model(self, n_unit_lstm=200, n_unit_atten=400):
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
        else:
            length = self._win_len

        class CSIModel(nn.Module):
            def __init__(self, input_dim=52, hidden_dim=n_unit_lstm, atten_dim=n_unit_atten, num_classes=7):
                super(CSIModel, self).__init__()
                self.bilstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    bidirectional=True,
                    batch_first=True
                )
                self.attention = AttenLayer(atten_dim)
                self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2因为双向LSTM
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                # x shape: (batch, seq_len, 52)
                lstm_out, _ = self.bilstm(x)  # (batch, seq_len, 2*hidden_dim)
                atten_out = self.attention(lstm_out)  # (batch, 2*hidden_dim)
                logits = self.fc(atten_out)  # (batch, 7)
                return self.softmax(logits)

        model = CSIModel().to(self.device)
        return model

    @staticmethod
    def load_model(pth_path):
        model = torch.load(pth_path)
        return model


# 训练和验证函数
def train_model(model, train_loader, val_loader, config, epochs=200, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, torch.argmax(y, dim=1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == torch.argmax(y, dim=1)).sum().item()

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                outputs = model(x)
                loss = criterion(outputs, torch.argmax(y, dim=1))

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == torch.argmax(y, dim=1)).sum().item()

        # 计算指标
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {history["loss"][-1]:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {history["val_loss"][-1]:.4f}, Val Acc: {val_acc:.4f}\n')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, 'best_atten.pth')

    return history


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Error! Correct Command: python3 csimodel_pytorch.py Dataset_folder_path")
        sys.exit(1)
    raw_data_folder = sys.argv[1]

    # 配置和预处理
    cfg = CSIModelConfig(win_len=500, step=50, thrshd=0.6, downsample=1)
    numpy_tuple = cfg.preprocessing(raw_data_folder, save=True)

    # 解析数据
    (x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend,
     x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk) = numpy_tuple

    # 合并所有数据并划分训练集和验证集
    all_x = np.concatenate([x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk], axis=0)
    all_y = np.concatenate([y_lie_down, y_fall, y_bend, y_run, y_sitdown, y_standup, y_walk], axis=0)

    # 随机划分
    dataset = CSIDataset(all_x, all_y)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(379))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 构建和训练模型
    model = cfg.build_model(n_unit_lstm=200, n_unit_atten=400)
    history = train_model(model, train_loader, val_loader, cfg, epochs=200)

    # 加载最佳模型并评估
    best_model = CSIModelConfig.load_model('best_atten.pth')
    best_model.eval()

    # 生成混淆矩阵
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(cfg.device)
            outputs = best_model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(y, dim=1).numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmd = ConfusionMatrixDisplay(cm, display_labels=cfg._labels)
    plt.figure(figsize=(12, 10))
    cmd.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # 保存日志
    with open('log.txt', 'w') as f:
        f.write('Training Log:\n')
        f.write(f'Best Validation Accuracy: {max(history["val_accuracy"]):.4f}\n')
        f.write('Epochs vs Accuracy:\n')
        for i in range(len(history["accuracy"])):
            f.write(f'Epoch {i + 1}: Train {history["accuracy"][i]:.4f}, Val {history["val_accuracy"][i]:.4f}\n')


