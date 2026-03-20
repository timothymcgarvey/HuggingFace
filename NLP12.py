import torch
print(torch.backends.mps.is_available(), torch.backends.mps.is_built())

import torch
print(torch.__version__)
import accelerate
print(accelerate.__version__)
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="bf16")

print(accelerator.state.mixed_precision)
print(accelerator.device)

