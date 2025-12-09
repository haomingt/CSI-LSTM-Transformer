import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for data, labels in tqdm(dataloader, desc="Eval", leave=False):
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        _, preds = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        if criterion:
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)

    acc = correct / total
    avg_loss = loss_sum / total if criterion else None

    return avg_loss, acc
