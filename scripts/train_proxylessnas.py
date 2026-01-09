import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
from dataset import OwlSoundDataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# torchaudio.set_audio_backend("soundfile")

# === Configuration ===
DATA_DIR = "../buowset1.1"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
MODEL_PATH = "../models/buowset1.1/proxylessnas_owl.pth"
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Load metadata ===
metadata = pd.read_csv(META_FILE)
train_df = metadata[metadata["fold"].isin([0, 1, 2])].reset_index(drop=True)
val_df = metadata[metadata["fold"] == 3].reset_index(drop=True)


# === Datasets and DataLoaders ===
train_dataset = OwlSoundDataset(train_df, AUDIO_DIR)
val_dataset = OwlSoundDataset(val_df, AUDIO_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# === Load ProxylessNAS model ===
model = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_mobile', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(DEVICE)


# === Loss, Optimizer, and Tracking ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Metrics tracking ===
train_losses, val_losses = [], []
val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], []
best_accuracy = 0.0
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# === Training loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc = correct / total
    val_accuracies.append(val_acc)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Best model saved with accuracy: {best_accuracy:.4f}")

# === Save CSV and Plots ===
metrics_df = pd.DataFrame({
    "epoch": list(range(1, NUM_EPOCHS + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies,
    "val_precision": val_precisions,
    "val_recall": val_recalls,
    "val_f1": val_f1s
})
metrics_df.to_csv("../graphs/proxylessnas_training_metrics.csv", index=False)

plt.figure()
plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss")
plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("../graphs/proxylessnas_train_loss.png")

plt.figure()
plt.plot(metrics_df["epoch"], metrics_df["val_accuracy"], label="Accuracy")
plt.plot(metrics_df["epoch"], metrics_df["val_precision"], label="Precision")
plt.plot(metrics_df["epoch"], metrics_df["val_recall"], label="Recall")
plt.plot(metrics_df["epoch"], metrics_df["val_f1"], label="F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics")
plt.legend()
plt.savefig("../graphs/proxylessnas_metrics.png")

print("Training complete. Metrics saved to CSV and PNGs.")
