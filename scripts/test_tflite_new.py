import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import OwlSoundDataset
from tqdm import tqdm
import time
from ai_edge_litert.interpreter import Interpreter
import numpy as np

# --- Config & Paths ---
DATA_DIR = "/Users/ryanhoang/Downloads/buowset1.33"
# AUDIO_DIR = os.path.join(DATA_DIR, "audio")
AUDIO_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/TEST_SET/label_0"
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
BATCH_SIZE = 1
NUM_CLASSES = 6
FOLD = 4
LABEL = 0
GRAPHS_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/something"

os.makedirs(GRAPHS_DIR, exist_ok=True)

# --- Load metadata and dataset ---
metadata = pd.read_csv(META_FILE)
test_df = metadata[metadata["fold"] == FOLD].reset_index(drop=True)
test_df = test_df[test_df["label"] == LABEL].reset_index(drop=True)
test_dataset = OwlSoundDataset(test_df, AUDIO_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Paths Dictionary ---
models_dict = {
    "BASELINE": '/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/mbconv/buow_tinycnn_mbconv2_100_melfix.tflite',
    "INT16": '/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/seed_experiment/1/buow_model_int16_proportional.tflite',
    "INT8": '/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/seed_experiment/1/buow_model_int8_proportional.tflite'
}

# Dictionary to hold all extracted metrics for plotting
results = {}


# --- Evaluation Function ---
def evaluate(model_path, name):
    print(f"\n--- Starting Evaluation: {name} ---")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    file_size_kb = os.path.getsize(model_path) / 1024

    preds, labels, latencies = [], [], []
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for x, y in tqdm(test_loader, desc=f"Evaluating {name}"):
        if (x.shape[3] != input_details[0]['shape'][3]):
            continue

        interpreter.set_tensor(input_details[0]['index'], x)

        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

        out = interpreter.get_tensor(output_details[0]['index'])
        preds.extend(np.argmax(out, axis=1))
        labels.extend(y.numpy() if isinstance(y, torch.Tensor) else y)

    acc = sum([p == l for p, l in zip(preds, labels)]) / len(labels)
    avg_latency = np.mean(latencies)
    cm = confusion_matrix(labels, preds)
    report_dict = classification_report(labels, preds, output_dict=True, zero_division=0)

    # We now save the raw predictions and labels to use for Bootstrapping later
    results[name] = {
        "accuracy": acc,
        "size_kb": file_size_kb,
        "latency_ms": avg_latency,
        "confusion_matrix": cm,
        "class_reports": report_dict,
        "raw_preds": np.array(preds),
        "raw_labels": np.array(labels)
    }

    print(f"{name} Results -> Accuracy: {acc:.4f} | Size: {file_size_kb:.2f} KB | Avg Latency: {avg_latency:.2f} ms")


# --- Run Evaluations ---
for name, path in models_dict.items():
    evaluate(path, name)

# ==========================================
# --- GENERATE PRESENTATION GRAPHS ---
# ==========================================
print("\nGenerating Presentation Plots...")

names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]
sizes = [results[n]["size_kb"] for n in names]
latencies = [results[n]["latency_ms"] for n in names]

sns.set_theme(style="whitegrid")
palette = ["#4C72B0", "#55A868", "#C44E52"]

# --- 1. The Motivation: Model Memory Footprint ---
plt.figure(figsize=(6, 4))
sns.barplot(x=names, y=sizes, hue=names, palette=palette, legend=False)
plt.ylabel("Model Size (KB)")
plt.title("Hardware Constraint: Memory Footprint")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "1_motivation_memory_footprint.png"), dpi=300)
plt.show()
plt.close()

# --- 2. The Motivation: Inference Latency ---
plt.figure(figsize=(6, 4))
sns.barplot(x=names, y=latencies, hue=names, palette=palette, legend=False)
plt.ylabel("Avg Inference Latency (ms)")
plt.title("Hardware Constraint: Processing Speed")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "2_motivation_latency.png"), dpi=300)
plt.show()
plt.close()

# --- 3. The Core Trade-off: Pareto Frontier ---
plt.figure(figsize=(8, 5))
plt.scatter(sizes, accuracies, c=palette, s=150, zorder=5, edgecolor='black')
plt.plot(sizes, accuracies, linestyle='--', color='gray', alpha=0.5, zorder=1)

for i, name in enumerate(names):
    plt.annotate(name, (sizes[i], accuracies[i]), xytext=(8, 8), textcoords='offset points', fontweight='bold')

plt.title("The Efficiency vs. Efficacy Trade-off")
plt.xlabel("Model Size (KB) -> (Smaller is better)")
plt.ylabel("Overall Accuracy -> (Higher is better)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "3_tradeoff_pareto_frontier.png"), dpi=300)
plt.show()
plt.close()

# --- 4. The Nuance: Class-Wise F1-Score Degradation (UPDATED) ---

# Dictionary keys must be the string numbers so c_id can look them up
CLASS_MAPPING = {
    '0': 'cluck',
    '1': 'coocoo',
    '2': 'twitter',
    '3': 'alarm',
    '4': 'chick begging',
    '5': 'no_buow'
}

class_ids = [str(i) for i in range(NUM_CLASSES)]
f1_data = []
N_BOOTSTRAPS = 100

print("Bootstrapping test set to calculate F1-Score variance...")
for name in names:
    preds = results[name]["raw_preds"]
    labels = results[name]["raw_labels"]
    n_samples = len(labels)

    for _ in tqdm(range(N_BOOTSTRAPS), desc=f"Bootstrapping {name}"):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_preds = preds[indices]
        sample_labels = labels[indices]

        report = classification_report(sample_labels, sample_preds, output_dict=True, zero_division=0)

        for c_id in class_ids:
            if c_id in report:
                f1_data.append({
                    "Precision": name,
                    "Audio Class": CLASS_MAPPING.get(c_id, f"Class {c_id}"),  # Using the mapping here
                    "F1-Score": report[c_id]["f1-score"]
                })

f1_df = pd.DataFrame(f1_data)

plt.figure(figsize=(12, 6))  # Made slightly wider to accommodate annotations

# Plot using standard deviation ('sd') instead of confidence intervals ('ci')
ax = sns.barplot(
    data=f1_df,
    x="Audio Class",
    y="F1-Score",
    hue="Precision",
    palette=palette,
    errorbar='sd',
    capsize=0.1,
    errwidth=1.5
)

# --- Add Delta Annotations ---
# Seaborn orders ax.containers by the hue order (0: BASELINE, 1: INT16, 2: INT8)
baseline_heights = [rect.get_height() for rect in ax.containers[0]]

for i, container in enumerate(ax.containers):
    if i == 0:
        continue  # Skip labeling the baseline with a delta

    for j, rect in enumerate(container):
        height = rect.get_height()
        baseline = baseline_heights[j]

        # Avoid annotating empty bars
        if pd.isna(height) or pd.isna(baseline) or height < 0.01:
            continue

        delta = height - baseline

        # Color the text red if the F1-score dropped, green if it improved/stayed same
        text_color = '#C44E52' if delta < 0 else '#55A868'

        ax.annotate(f'{delta:+.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset above the error bar
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=text_color)

plt.title("Nuance: Which classes break under quantization? (Mean ± 1 SD)")
plt.ylim(0, 1.15)  # Increased upper limit slightly so annotations don't get cut off
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "4_nuance_class_f1_scores_annotated.png"), dpi=300)
plt.show()
plt.close()

# --- 5. Delta Confusion Matrix ---
if "BASELINE" in results and "INT8" in results:
    delta_cm = results["INT8"]["confusion_matrix"] - results["BASELINE"]["confusion_matrix"]

    plt.figure(figsize=(8, 6))

    # Create an ordered list of class names for the axes
    axis_labels = [CLASS_MAPPING[str(i)] for i in range(NUM_CLASSES)]

    sns.heatmap(delta_cm, annot=True, fmt="d", cmap="RdBu_r", center=0,
                cbar_kws={'label': 'Change in Prediction Count'},
                xticklabels=axis_labels, yticklabels=axis_labels)

    plt.title("Delta Matrix: INT8 Errors relative to Baseline\n(Red = New Errors, Blue = Fewer Errors)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "5_nuance_delta_matrix.png"), dpi=300)
    plt.show()
    plt.close()

print(f"\nAll presentation plots successfully saved to:\n{GRAPHS_DIR}")