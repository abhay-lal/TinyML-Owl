import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import tensorflow as tf
import nobuco
from nobuco import ChannelOrder

# === 1. Configuration & Paths ===
# Ensure these point to your server paths
OUTPUT_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/proxylessnas_1.33"
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

H = 128
W = 241
C_DEPLOY = 3
NUM_CLASSES = 6

# Path to the .pth file you just finished training
WEIGHTS_PATH = "/Users/ryanhoang/Desktop/OWL STUFF/models/proxylessnas_1.33" # Update if you named it differently
MODEL_NAME = "proxylessnas"

OUT_TFLITE = os.path.join(OUTPUT_DIR, MODEL_NAME + ".tflite")
saved_model_dir = os.path.join(OUTPUT_DIR, MODEL_NAME + "_saved_model")

# === 2. Load the PyTorch Model ===
print("Loading PyTorch ProxylessNAS...")
pt_model = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_mobile', pretrained=False)
pt_model.classifier = nn.Linear(pt_model.classifier.in_features, NUM_CLASSES)

state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
pt_model.load_state_dict(state_dict)
pt_model.eval()
print("Successfully loaded weights.")

# === 3. The Deploy Wrapper ===
# ProxylessNAS uses 3 channels natively, so we do NOT slice it down to 1 channel
# like the old TinyAudioCNN script did.
class DeployWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor):
        return self.core(x)

wrapped = DeployWrapper(pt_model).eval()
dummy_input = torch.randn(1, C_DEPLOY, H, W, dtype=torch.float32)

# === 4. Convert to Keras (Nobuco) ===
print("Starting Nobuco PyTorch -> Keras conversion...")
keras_model = nobuco.pytorch_to_keras(
    wrapped,
    args=[dummy_input],
    kwargs=None,
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH,
)

keras_model.save(saved_model_dir)
print(f"Saved intermediate Keras model to {saved_model_dir}")

# === 5. Convert to TFLite ===
print("Starting Keras -> TFLite conversion...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# This optimization flag triggers "Hybrid Quantization" (weights are 8-bit, activations are float32)
# If you plan to run the Proportional PTQ script later, this line just sets up the baseline.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print("Wrote TFLite file:", OUT_TFLITE)

# === 6. Verify and Export Header ===
interp = tf.lite.Interpreter(model_path=OUT_TFLITE)
interp.allocate_tensors()

in0 = interp.get_input_details()[0]
out0 = interp.get_output_details()[0]

print("TFLite input shape:", in0["shape"], "dtype:", in0["dtype"])
print("TFLite output shape:", out0["shape"], "dtype:", out0["dtype"])

# Convert to C Header for the microcontroller
header_path = f"{os.path.splitext(OUT_TFLITE)[0]}.h"
os.system(f'xxd -i "{OUT_TFLITE}" > "{header_path}"')
print("Created header:", header_path)