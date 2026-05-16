import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from ai_edge_litert.interpreter import Interpreter
import numpy as np

# --- Config ---
DATA_DIR = "/Users/ryanhoang/Downloads/buowset1.33"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
# AUDIO_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/TEST_SET/label_0"
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
BATCH_SIZE = 1
NUM_CLASSES = 6
FOLD = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
os.makedirs("/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/graphs", exist_ok=True)

# --- Load metadata and dataset ---
metadata = pd.read_csv(META_FILE)
test_df = metadata[metadata["fold"] == FOLD].reset_index(drop=True)
test_dataset = OwlSoundDataset(test_df, AUDIO_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Evaluation function ---
accuracies = {}


def evaluate(interpreter, name):
    preds, labels = [], []
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for x, y in tqdm(test_loader, desc=f"Evaluating {name}"):
        if (x.shape[3] != input_details[0]['shape'][3]):
            print('Warning: skipping input due to shape mismatch in time dimension')
            continue
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        p = np.argmax(out, axis=1)
        preds.extend(p)
        labels.extend(y)

    print(labels)
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


NEW_interpreter = Interpreter(
    model_path='/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/mbconv/buow_tinycnn_mbconv2_100_melfix.tflite')
NEW_interpreter.allocate_tensors()
evaluate(NEW_interpreter, "BASELINE")

Baseline_interpreter = Interpreter(
    model_path='/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/seed_experiment/1/buow_model_int16_proportional.tflite')
Baseline_interpreter.allocate_tensors()
evaluate(Baseline_interpreter, "INT16")

Proportional_interpreter = Interpreter(
    model_path='/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/seed_experiment/1/buow_model_int8_proportional.tflite')
Proportional_interpreter.allocate_tensors()
evaluate(Proportional_interpreter, "INT8")

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
