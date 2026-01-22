import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from dataset import OwlSoundDataset, OwlSoundWaveformDataset
from tqdm import tqdm
from efficientleaf.efficientleaf import EfficientLeaf
import os

from torch import nn
from torch import Tensor
from typing import Optional
from functools import partial

class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)

class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x
    
class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features, 
                                      expanded_features, 
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )

# Lightweight CNN
class TinyAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.frontend = EfficientLeaf(n_filters=128, min_freq=60, max_freq=7800,
                                sample_rate=16000,
                                num_groups=8, conv_win_factor=6, stride_factor=16)
        
        self.net = nn.Sequential(
            FusedMBConv(1, 8),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,8,32,129)

            FusedMBConv(8, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,16,16,64)

            FusedMBConv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        filters = self.frontend(x) # [B, 2, n_filters, time]
        filters = filters[:, 1:, :, :] # Take the median filtered channel
        return self.net(filters)

class TinyAudioCNN_MBConv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.frontend = EfficientLeaf(n_filters=128, min_freq=60, max_freq=7800,
                                sample_rate=16000,
                                num_groups=8, conv_win_factor=6, stride_factor=16)
        
        self.net = nn.Sequential(
            FusedMBConv(1, 8),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,8,32,129)

            FusedMBConv(8, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,16,16,64)

            FusedMBConv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        filters = self.frontend(x) # [B, 2, n_filters, time]
        filters = filters[:, 1:, :, :] # Take the median filtered channel
        print(f'Filter shape: {filters.shape}')
        return self.net(filters)


# --- Config ---
DATA_DIR = "../buowset1.1"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
BATCH_SIZE = 128
NUM_CLASSES = 6
FOLD = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
os.makedirs("../graphs", exist_ok=True)

# --- Load metadata and dataset ---
metadata = pd.read_csv(META_FILE)
test_df = metadata[metadata["fold"] == FOLD].reset_index(drop=True)
test_dataset = OwlSoundWaveformDataset(test_df, AUDIO_DIR, channels=3)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Evaluation function ---
accuracies = {}

def evaluate(model, name):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Evaluating {name}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            p = torch.argmax(out, axis=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = sum([p == l for p, l in zip(preds, labels)]) / len(labels)
    accuracies[name] = acc

    print(f"\n{name} Accuracy: {acc:.4f}")
    report_dict = classification_report(labels, preds, output_dict=True)
    print(classification_report(labels, preds))

    # Save classification report to CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"../graphs/{name}_classification_report.csv")

    # Save confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"../graphs/{name}_confusion_matrix.png")
    plt.show()

# tinycnn = TinyAudioCNN(NUM_CLASSES)
# tinycnn.load_state_dict(torch.load("../models/buowset1.1/buow_tinycnn.pth"))
# tinycnn.to(DEVICE)
# evaluate(tinycnn, 'TinyCNN_new')

tinycnn_mbconv = TinyAudioCNN_MBConv(NUM_CLASSES)
tinycnn_mbconv.load_state_dict(torch.load("../models/buowset1.1/buow_tinycnn_leaf.pth"))
tinycnn_mbconv.to(DEVICE)
evaluate(tinycnn_mbconv, 'TinyCNN_Leaf')

# --- Evaluate MobileNetV2 ---
# mobilenet = models.mobilenet_v2(pretrained=False)
# mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, NUM_CLASSES)
# mobilenet.load_state_dict(torch.load("../models/buowset1.1/mobilenetv2_owl.pth", map_location=DEVICE))
# mobilenet.to(DEVICE)
# evaluate(mobilenet, "MobileNetV2")

# # # --- Evaluate ProxylessNAS ---
# proxyless = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_mobile', pretrained=True)
# proxyless.classifier = nn.Linear(proxyless.classifier.in_features, NUM_CLASSES)
# proxyless.load_state_dict(torch.load("../models/buowset1.1/proxylessnas_owl.pth", map_location=DEVICE))
# proxyless.to(DEVICE)
# evaluate(proxyless, "ProxylessNAS")

# --- Accuracy Comparison Plot ---
plt.figure(figsize=(6, 4))
plt.bar(accuracies.keys(), accuracies.values(), color=["skyblue", "salmon"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Test Accuracy Comparison")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("../graphs/test_accuracy_comparison.png")
plt.show()
