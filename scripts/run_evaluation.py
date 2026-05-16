import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
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
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")
BATCH_SIZE = 1
NUM_CLASSES = 6
FOLD = 4

# --- New: Explicit Class Mapping ---
# Maps the integer output of the model to the actual string names
CLASS_NAMES = {
    0: "Cluck",
    1: "Coocoo",
    2: "Twitter",
    3: "Alarm",
    4: "Chick Begging",
    5: "No Buow"
}

# Change these paths to exactly where your models and folders live
ROOT_EXPERIMENT_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/seed_experiment"
BASELINE_MODEL = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/mbconv/buow_tinycnn_mbconv2_100_melfix.tflite"

# --- Load metadata and dataset ---
metadata = pd.read_csv(META_FILE)
test_df = metadata[metadata["fold"] == FOLD].reset_index(drop=True)
test_dataset = OwlSoundDataset(test_df, AUDIO_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate(model_path, name, desc_suffix=""):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    file_size_kb = os.path.getsize(model_path) / 1024

    preds, labels, latencies = [], [], []
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for x, y in tqdm(test_loader, desc=f"Evaluating {name} {desc_suffix}", leave=False):
        # Validate channel dimensions
        if x.shape[3] != input_details[0]['shape'][3]:
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

    # Extract Overall F1 Score (Macro Average)
    overall_f1 = report_dict["macro avg"]["f1-score"]

    return {
        "accuracy": acc,
        "size_kb": file_size_kb,
        "latency_ms": avg_latency,
        "confusion_matrix": cm,
        "class_reports": report_dict,
        "overall_f1": overall_f1
    }


# --- 1. Evaluate Baseline (Only needs to happen once) ---
print("\n--- Evaluating BASELINE (Float32) ---")
base_res = evaluate(BASELINE_MODEL, "BASELINE")
print(
    f"BASELINE -> Acc: {base_res['accuracy']:.4f} | Size: {base_res['size_kb']:.2f} KB | Overall F1: {base_res['overall_f1']:.4f}")

# Standardized colors for all graphs
palette = {"BASELINE": "#4C72B0", "INT16": "#55A868", "INT8": "#C44E52"}

# --- 2. Iterate Through Seeds, Evaluate, and Generate Plots per Seed ---
for seed in range(1, 21):
    print(f"\n======================================")
    print(f"--- Processing Seed {seed}/20 ---")
    print(f"======================================")

    seed_dir = os.path.join(ROOT_EXPERIMENT_DIR, str(seed))
    int8_path = os.path.join(seed_dir, "buow_model_int8_proportional.tflite")
    int16_path = os.path.join(seed_dir, "buow_model_int16_proportional.tflite")

    # Create a specific directory for this seed's graphs
    seed_graphs_dir = os.path.join(seed_dir, "graphs")
    os.makedirs(seed_graphs_dir, exist_ok=True)

    seed_results = {"BASELINE": base_res}

    if os.path.exists(int8_path):
        seed_results["INT8"] = evaluate(int8_path, "INT8", f"(Seed {seed})")
    if os.path.exists(int16_path):
        seed_results["INT16"] = evaluate(int16_path, "INT16", f"(Seed {seed})")

    # Ensure we have all models before plotting
    if "INT8" not in seed_results or "INT16" not in seed_results:
        print(f"Skipping plot generation for Seed {seed} due to missing models.")
        continue

    # Prepare DataFrame for this specific seed
    plot_data = []
    for m_type, res in seed_results.items():
        plot_data.append({
            "Model": m_type,
            "Accuracy": res["accuracy"],
            "Size (KB)": res["size_kb"],
            "Latency (ms)": res["latency_ms"],
            "Overall F1": res["overall_f1"]
        })
    df_metrics = pd.DataFrame(plot_data)
    sns.set_theme(style="whitegrid")

    # --- Graph 1. Model Memory Footprint ---
    plt.figure(figsize=(7, 5))
    ax1 = sns.barplot(data=df_metrics, x="Model", y="Size (KB)", palette=palette)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f KB', padding=3, fontweight='bold')
    plt.title(f"Memory Footprint (Seed {seed})\nBaseline vs. Quantized")
    plt.ylabel("Model Size (KB)")
    plt.ylim(0, df_metrics["Size (KB)"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(seed_graphs_dir, "1_memory_footprint.png"), dpi=300)
    plt.close()

    # --- Graph 2. Inference Latency ---
    plt.figure(figsize=(7, 5))
    ax2 = sns.barplot(data=df_metrics, x="Model", y="Latency (ms)", palette=palette)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2f ms', padding=3, fontweight='bold')
    plt.title(f"Inference Latency (Seed {seed})\nBaseline vs. Quantized")
    plt.ylabel("Avg Inference Latency (ms)")
    plt.ylim(0, df_metrics["Latency (ms)"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(seed_graphs_dir, "2_inference_latency.png"), dpi=300)
    plt.close()

    # --- Graph 3. The Core Trade-off: Pareto Frontier ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_metrics, x="Size (KB)", y="Accuracy", hue="Model", palette=palette, s=200,
                    edgecolor='black', zorder=5)

    for i in range(len(df_metrics)):
        plt.annotate(
            f"{df_metrics['Model'][i]}\nAcc: {df_metrics['Accuracy'][i]:.3f}\nF1: {df_metrics['Overall F1'][i]:.3f}",
            (df_metrics['Size (KB)'][i], df_metrics['Accuracy'][i]),
            xytext=(10, 10), textcoords='offset points', fontweight='bold'
        )

    plt.title(f"Efficiency vs. Efficacy Trade-off (Seed {seed})")
    plt.xlabel("Model Size (KB) -> (Smaller is better)")
    plt.ylabel("Overall Accuracy -> (Higher is better)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(seed_graphs_dir, "3_tradeoff_pareto_frontier.png"), dpi=300)
    plt.close()

    # --- Graph 4. Class-Wise F1-Score (Updated with string labels) ---
    f1_data = []
    for m_type in ["BASELINE", "INT16", "INT8"]:
        res = seed_results[m_type]
        for c_id in [str(i) for i in range(NUM_CLASSES)]:
            if c_id in res["class_reports"]:
                f1_data.append({
                    "Model": m_type,
                    "Audio Class": CLASS_NAMES[int(c_id)],  # <-- Injects the string name here
                    "F1-Score": res["class_reports"][c_id]["f1-score"]
                })

    f1_df = pd.DataFrame(f1_data)

    plt.figure(figsize=(12, 6))
    ax4 = sns.barplot(data=f1_df, x="Audio Class", y="F1-Score", hue="Model", palette=palette)

    # Label every bar with its F1 score value
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.3f', padding=3, rotation=90, size=9)

    # --- Adding the Overall F1 Score at the Top ---
    base_f1 = seed_results["BASELINE"]["overall_f1"]
    int16_f1 = seed_results["INT16"]["overall_f1"]
    int8_f1 = seed_results["INT8"]["overall_f1"]

    overall_f1_text = f"Overall F1 Scores\nBASELINE: {base_f1:.3f}   |   INT16: {int16_f1:.3f}   |   INT8: {int8_f1:.3f}"

    # Place text box inside the plot at the very top center
    plt.text(0.5, 0.95, overall_f1_text, transform=ax4.transAxes,
             fontsize=11, fontweight='bold', ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.title(f"Nuance: Which classes break under quantization? (Seed {seed})")
    plt.ylim(0, 1.25)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(seed_graphs_dir, "4_class_f1_scores.png"), dpi=300)
    plt.close()

    # --- Graph 5. Delta Confusion Matrices (Updated with string labels) ---
    base_cm = seed_results["BASELINE"]["confusion_matrix"]
    delta_int16 = seed_results["INT16"]["confusion_matrix"] - base_cm
    delta_int8 = seed_results["INT8"]["confusion_matrix"] - base_cm

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Create an ordered list of class names for the axes
    axis_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    sns.heatmap(delta_int16, annot=True, fmt="d", cmap="RdBu_r", center=0, cbar_kws={'label': 'Change in Count'},
                xticklabels=axis_labels, yticklabels=axis_labels, ax=axes[0])  # <-- Added tick labels
    axes[0].set_title(f"INT16 Delta vs Baseline (Seed {seed})\n(Red = More Errors, Blue = Fewer Errors)")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("True Class")
    axes[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels so they don't overlap

    sns.heatmap(delta_int8, annot=True, fmt="d", cmap="RdBu_r", center=0, cbar_kws={'label': 'Change in Count'},
                xticklabels=axis_labels, yticklabels=axis_labels, ax=axes[1])  # <-- Added tick labels
    axes[1].set_title(f"INT8 Delta vs Baseline (Seed {seed})\n(Red = More Errors, Blue = Fewer Errors)")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("True Class")
    axes[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels so they don't overlap

    plt.tight_layout()
    plt.savefig(os.path.join(seed_graphs_dir, "5_delta_confusion_matrices.png"), dpi=300)
    plt.close()

    print(f"Plots successfully saved for Seed {seed} in: {seed_graphs_dir}")

print("\nAll seeds processed successfully!")