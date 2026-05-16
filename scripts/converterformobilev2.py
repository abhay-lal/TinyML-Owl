import os
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import nobuco
from nobuco import ChannelOrder
from torchvision import models

# === Configuration ===
# Update these paths to match your server locations
MODEL_NAME = "mobilenetv2_owl"
WEIGHTS_PATH = "/Users/ryanhoang/Desktop/OWL STUFF/models/mobilenetv2_1.33" # Path from your training script
OUTPUT_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/mobilenetv2_1.33"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input dimensions - Ensure these match your OwlSoundDataset output
# Standard MobileNetV2 usually expects 3 channels (RGB)
H, W = 128, 241
C_DEPLOY = 3
NUM_CLASSES = 6

# === 1. Load the PyTorch Model ===
# We recreate the exact architecture from your training script
pt_model = models.mobilenet_v2(weights=None) # No need for pretrained here
pt_model.classifier[1] = nn.Linear(pt_model.last_channel, NUM_CLASSES)

# Load your trained weights
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
pt_model.load_state_dict(state_dict)
pt_model.eval()
print(f"Successfully loaded MobileNetV2 weights from {WEIGHTS_PATH}")

# === 2. Convert to Keras via Nobuco ===
# We wrap the model to ensure it handles the channel order correctly for TFLite
class DeployWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor):
        return self.core(x)

wrapped = DeployWrapper(pt_model).eval()
dummy_input = torch.randn(1, C_DEPLOY, H, W, dtype=torch.float32)

keras_model = nobuco.pytorch_to_keras(
    wrapped,
    args=[dummy_input],
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH,
)

# === 3. TFLite Conversion & Optimization ===
saved_model_dir = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_saved_model")
keras_model.save(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# This provides basic weight quantization (Hybrid)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TFLite file
OUT_TFLITE = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.tflite")
with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print(f"Wrote TFLite model to: {OUT_TFLITE}")

# === 4. Export to C Header (.h) for Microcontrollers ===
os.system(f'xxd -i "{OUT_TFLITE}" > "{os.path.splitext(OUT_TFLITE)[0]}.h"')
print(f"Created C header file: {os.path.splitext(OUT_TFLITE)[0]}.h")