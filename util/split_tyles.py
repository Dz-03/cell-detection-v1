import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_folder  = os.path.join(BASE_DIR, "../cell_images_jpg")
output_folder = os.path.join(BASE_DIR, "../cell_tiles")
os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder)
         if f.lower().endswith('.jpg')]

print(f"Найдено файлов: {len(files)}")

for file in files:
    input_path = os.path.join(input_folder, file)
    img = Image.open(input_path)
    w, h = img.size

    mid_w = w // 2
    mid_h = h // 2

    tiles = {
        "tl": img.crop((0,     0,     mid_w, mid_h)),
        "tr": img.crop((mid_w, 0,     w,     mid_h)),
        "bl": img.crop((0,     mid_h, mid_w, h    )),
        "br": img.crop((mid_w, mid_h, w,     h    )),
    }

    name = os.path.splitext(file)[0]
    for tile_name, tile in tiles.items():
        out_path = os.path.join(output_folder, f"{name}_{tile_name}.jpg")
        tile.save(out_path, "JPEG", quality=95)
        print(f"✅ {name}_{tile_name}.jpg | размер: {tile.size}")

print(f"\n=== Итого ===")
print(f"Входных снимков: {len(files)}")
print(f"Выходных тайлов: {len(files) * 4}")