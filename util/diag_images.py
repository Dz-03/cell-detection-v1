import os
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(BASE_DIR, "../cell_images")

files = [f for f in os.listdir(input_folder)
         if f.lower().endswith(('.tif', '.tiff'))]

for file in files[:3]:
    path = os.path.join(input_folder, file)
    img = Image.open(path)
    arr = np.array(img)

    print(f"Файл:        {file}")
    print(f"Режим:       {img.mode}")
    print(f"dtype:       {arr.dtype}")
    print(f"Min значение: {arr.min()}")
    print(f"Max значение: {arr.max()}")
    print("---")


