from TinyCNN import TinyAudioCNN_MBConv
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

NUM_CLASSES = 6
BATCH_SIZE = 128
NUM_FILTERS = 128
AUDIO_LENGTH_MS = 300

model = TinyAudioCNN_MBConv(NUM_CLASSES)
model.load_state_dict(torch.load('../models/buowset1.1/buow_tinycnn_mbconv.pth', weights_only=True, map_location=torch.device('cpu')))
model.eval()
example_inputs = (torch.randn(BATCH_SIZE, 1, NUM_FILTERS, AUDIO_LENGTH_MS),)

exported_program = export(model, example_inputs)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()]
).to_executorch()

with open("../models/executorch/buow_tinycnn_mbconv.pte", "wb") as file:
    file.write(executorch_program.buffer)

