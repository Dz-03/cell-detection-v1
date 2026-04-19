import os
import numpy as np
from PIL import Image

#Перевод изображений с tiff в jpg с нормализацей

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder  = os.path.join(BASE_DIR, "../cell_images_tiff")
output_folder = os.path.join(BASE_DIR, "../cell_images_jpg")
os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder)
         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))]

print(f"Найдено файлов: {len(files)}")

converted = 0
failed    = 0

for file in files:
    input_path  = os.path.join(input_folder, file)
    output_name = os.path.splitext(file)[0] + ".jpg"
    output_path = os.path.join(output_folder, output_name)

    try:
        img = Image.open(input_path)
        arr = np.array(img)

        # 16-bit изображение — нужна нормализация
        if arr.dtype == np.uint16:
            print(f"⚠️  {file} — 16-bit, нормализую...")

            # Нормализация: растягиваем реальный диапазон на 0–255
            arr_min = arr.min()
            arr_max = arr.max()

            if arr_max > arr_min:  # защита от деления на ноль
                arr = (arr - arr_min) / (arr_max - arr_min) * 255
            else:
                arr = np.zeros_like(arr)

            arr = arr.astype(np.uint8)
            img = Image.fromarray(arr)

        # Конвертируем в RGB
        img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=95)
        print(f"✅ {file} → {output_name}")
        converted += 1

    except Exception as e:
        print(f"❌ {file}: ошибка — {e}")
        failed += 1

print(f"\n=== Готово ===")
print(f"Конвертировано: {converted}")
print(f"Ошибок:         {failed}")