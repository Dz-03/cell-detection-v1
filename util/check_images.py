import cv2
import os
from PIL import Image

img_folder = r"C:\Users\Admin\PycharmProjects\cell-detection\cell_images"

img_files = [f for f in os.listdir(img_folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))]

print(f"Найдено файлов: {len(img_files)}")
print("---")

for img_file in img_files[:5]:
    full_path = os.path.join(img_folder, img_file)

    # Сначала пробуем OpenCV
    img = cv2.imread(full_path)

    if img is not None:
        h, w = img.shape[:2]
        print(f"✅ {img_file}: {w} x {h} px (OpenCV)")
    else:
        # Если OpenCV не справился — пробуем Pillow
        try:
            pil_img = Image.open(full_path)
            w, h = pil_img.size
            print(f"✅ {img_file}: {w} x {h} px (Pillow) | формат: {pil_img.format}")
        except Exception as e:
            print(f" {img_file}: не удалось прочитать — {e}")