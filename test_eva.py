import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Ensure root directory is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.model_advanced import AdvancedMNISTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedMNISTCNN().to(device)

ckpt_path = os.path.join(BASE_DIR, "checkpoints", "best_advanced_model.pth")
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Normalization & Transforms
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

data_dir = os.path.join(BASE_DIR, "data")
full_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=eval_transform)
test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)

# 90% Train / 10% Validation Partition (seed=42)
val_size = int(len(full_train) * 0.1)      # 6,000
train_size = len(full_train) - val_size    # 54,000
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

print("\n" + "=" * 65)
print("             MNIST DATASET PARTITIONS & BREAKDOWN")
print("=" * 65)
print(f"  • Training Set   : {len(train_set):>6,d} samples (77.14% of total, 90.0% of pool)")
print(f"  • Validation Set : {len(val_set):>6,d} samples ( 8.57% of total, 10.0% of pool)")
print(f"  • Holdout Test   : {len(test_set):>6,d} samples (14.29% of total)")
print(f"  -------------------------------------------------------------")
print(f"  • Total Dataset  : {len(train_set) + len(val_set) + len(test_set):>6,d} samples (100.00% benchmark)")
print("=" * 65)

def evaluate(loader, name):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = (correct / total) * 100.0
    errs = total - correct
    return acc, errs, correct, total

val_acc, val_errs, val_corr, val_tot = evaluate(val_loader, "Validation Set")
test_acc, test_errs, test_corr, test_tot = evaluate(test_loader, "Test Set")

print(f"\n{'SPLIT':<18} | {'ACCURACY':<10} | {'CORRECT / TOTAL':<18} | {'ERRORS'}")
print("-" * 65)
print(f"{'Validation (6k)':<18} | {val_acc:>8.2f}% | {val_corr:>6,d} / {val_tot:>6,d}     | {val_errs} misclassifications")
print(f"{'Holdout Test (10k)':<18} | {test_acc:>8.2f}% | {test_corr:>6,d} / {test_tot:>6,d}     | {test_errs} misclassifications")
print("-" * 65)
print(f"  🏆 Generalization Gap (|Val - Test|): {abs(val_acc - test_acc):.2f}%")
print("=" * 65 + "\n")