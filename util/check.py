import torch
import cv2
import numpy as np
import albumentations
from ultralytics import YOLO

print("=== Проверка окружения ===")
print(f"PyTorch версия:        {torch.__version__}")
print(f"CUDA доступна:         {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Название GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:                  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ GPU не найден! Проверь установку CUDA")

print(f"OpenCV версия:         {cv2.__version__}")
print(f"NumPy версия:          {np.__version__}")
print("✅ Все библиотеки установлены корректно!")