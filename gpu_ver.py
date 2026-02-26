from accelerate import Accelerator
import torch

accelerator = Accelerator()
print("Accelerator device:", accelerator.device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
