import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from dataset import OwlSoundDataset
from tqdm import tqdm
import os

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
test_dataset = OwlSoundDataset(test_df, AUDIO_DIR)
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

# --- Evaluate MobileNetV2 ---
mobilenet = models.mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, NUM_CLASSES)
mobilenet.load_state_dict(torch.load("../models/buowset1.0/mobilenetv2_owl.pth", map_location=DEVICE))
mobilenet.to(DEVICE)
evaluate(mobilenet, "MobileNetV2")

# --- Evaluate ProxylessNAS ---
proxyless = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_mobile', pretrained=True)
proxyless.classifier = nn.Linear(proxyless.classifier.in_features, NUM_CLASSES)
proxyless.load_state_dict(torch.load("../models/buowset1.0/proxylessnas_owl.pth", map_location=DEVICE))
proxyless.to(DEVICE)
evaluate(proxyless, "ProxylessNAS")

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
