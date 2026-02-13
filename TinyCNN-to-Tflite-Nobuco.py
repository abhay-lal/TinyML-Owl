# When you run this code you will see that at the end there are a lot of non-converted ops,
# but after doing some research those non-converted ops are not actual TF ops, so it is fine.
# I am still looking into this to confirm there are no issues.

import os

# Nobuco Expects these settings
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import nobuco
from nobuco import ChannelOrder
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
        **kwargs,
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

    def forward(self, x):
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        return x + res


class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features, expanded_features, act=nn.ReLU6),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )


class TinyAudioCNN_MBConv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            FusedMBConv(1, 8),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            FusedMBConv(8, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            FusedMBConv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


H = 128
W = 241
C_DEPLOY = 3
NUM_CLASSES = 6

WEIGHTS_PATH = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-Owl-mbconv/models/buowset1.1/buow_tinycnn_mbconv.pth"  # Put something here pretty please (.pth)
OUT_TFLITE = (
    "buow_tinycnn_mbconv.tflite"  # Put something here pretty please if you want
)


pt_model = TinyAudioCNN_MBConv(num_classes=NUM_CLASSES)
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
pt_model.load_state_dict(state_dict)
pt_model.eval()
print("Loaded PyTorch TinyAudioCNN_MBConv.")


class DeployWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    # Training model has 1 channel, but for testing it should have 3 we are just taking one channel here.
    def forward(self, x_nchw3: torch.Tensor):
        x_nchw1 = x_nchw3[:, :1, :, :]
        return self.core(x_nchw1)


wrapped = DeployWrapper(pt_model).eval()
dummy_input = torch.randn(1, C_DEPLOY, H, W, dtype=torch.float32)

# First converting to a Keras/TF
keras_model = nobuco.pytorch_to_keras(
    wrapped,
    args=[dummy_input],
    kwargs=None,
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH,
)
print("Nobuco conversion done.")
print(
    "Keras input_shape:", keras_model.input_shape, "dtype:", keras_model.inputs[0].dtype
)


saved_model_dir = "../buow_tinycnn_mbconv"  # Put something here pretty please
keras_model.save(saved_model_dir)
print("SavedModel written to:", saved_model_dir)


# Then from Keras/TF to TFlite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


tflite_model = converter.convert()
with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print("Wrote:", OUT_TFLITE)


# Little Checker For Input Shape
interp = tf.lite.Interpreter(model_path=OUT_TFLITE)
interp.allocate_tensors()
in0 = interp.get_input_details()[0]
out0 = interp.get_output_details()[0]
print("TFLite input shape:", in0["shape"], "dtype:", in0["dtype"])
print("TFLite output shape:", out0["shape"], "dtype:", out0["dtype"])


# If you need it's here the .h file.
os.system(f"xxd -i {OUT_TFLITE} > {os.path.splitext(OUT_TFLITE)[0]}.h")
print("Created header:", f"{os.path.splitext(OUT_TFLITE)[0]}.h")
