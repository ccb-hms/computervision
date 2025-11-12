import os
from computervision.datasets import get_gpu_info
dev = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev)
device, device_str = get_gpu_info(device_number=dev)
print(f'Current device {device}')
print(f'Current device string {device_str}')
