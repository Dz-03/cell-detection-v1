from ultralytics import YOLO
from pathlib import Path
import glob
import pandas as pd
import json


def test_model():
    print("\n" + "=" * 60)
    print("ТЕСТ ДЕТЕКЦИИ КЛЕТОК")
    print("=" * 60 + "\n")

    # Находим модель
    # models = glob.glob('models/best.pt', recursive=True)
    models = glob.glob('models/yolov8m_optimized_6gb/weights/best.pt', recursive=True)

    if not models:
        print(" Модель не найдена!")
        return

    model_path = max(models, key=lambda p: Path(p).stat().st_mtime)
    print(f"✓ Модель: {model_path}\n")

    # Загружаем
    model = YOLO(model_path)

    # Находим тестовые изображения
    test_images = list(Path('cell_images_jpg').glob('*.jpg'))

    if not test_images:
        test_images = list(Path('cell-detection.yolov8/train/images').glob('*.jpg'))[:10]

    if not test_images:
        print(" Изображения не найдены!")
        return

    print(f"🖼  Найдено изображений для теста: {len(test_images)}\n")

    # Детекция
    print("⏳ Запуск детекции...\n")

    results = model.predict(
        source=test_images[:10],
        save=True,
        conf=0.25,  # Порог уверенности
        iou=0.45,
        device='cpu',
        project='runs/detect',
        name='test_results',
        verbose=False,  # Отключаем лишний вывод
    )

    # ==========================================
    # ВЫВОД КООРДИНАТ И СТАТИСТИКИ
    # ==========================================
    print("=" * 60)
    print(" РЕЗУЛЬТАТЫ ДЕТЕКЦИИ С КООРДИНАТАМИ")
    print("=" * 60 + "\n")

    total_detections = 0
    all_detections = []

    for i, result in enumerate(results):
        num_boxes = len(result.boxes)
        total_detections += num_boxes

        img_name = Path(result.path).name
        img_height, img_width = result.orig_shape

        print(f"Изображение {i + 1}: {img_name}")
        print(f"   Размер: {img_width} × {img_height} px")
        print(f"   Найдено клеток: {num_boxes}")

        if num_boxes == 0:
            print("Объекты не обнаружены\n")
            continue

        print()

        # Вывод координат каждого объекта
        for j, box in enumerate(result.boxes):
            # Координаты в разных форматах
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy, w, h = box.xywh[0].cpu().numpy()

            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = result.names[cls]

            # Сохраняем для JSON/CSV
            detection = {
                'image': img_name,
                'image_width': img_width,
                'image_height': img_height,
                'object_id': j + 1,
                'class': class_name,
                'confidence': round(conf, 4),
                'x1': round(float(x1), 2),
                'y1': round(float(y1), 2),
                'x2': round(float(x2), 2),
                'y2': round(float(y2), 2),
                'center_x': round(float(cx), 2),
                'center_y': round(float(cy), 2),
                'width': round(float(w), 2),
                'height': round(float(h), 2),
            }
            all_detections.append(detection)

            # Вывод в консоль
            print(f"   🔹 Клетка #{j + 1}")
            print(f"      Класс: {class_name}")
            print(f"      Уверенность: {conf:.1%}")
            print(f"      Координаты (x1, y1, x2, y2): ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            print(f"      Центр (cx, cy): ({cx:.1f}, {cy:.1f})")
            print(f"      Размер (w × h): {w:.1f} × {h:.1f} px")
            print()

        print("-" * 60 + "\n")

    # ==========================================
    # СОХРАНЕНИЕ КООРДИНАТ В ФАЙЛЫ
    # ==========================================
    output_dir = Path('runs/detect/test_results')
    output_dir.mkdir(parents=True, exist_ok=True)  # ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ

    # 1. Сохранение в JSON
    json_path = output_dir / 'detections.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)

    # 2. Сохранение в CSV
    csv_path = output_dir / 'detections.csv'
    df = pd.DataFrame(all_detections)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # 3. Сохранение в TXT (YOLO формат)
    txt_dir = output_dir / 'labels'
    txt_dir.mkdir(parents=True, exist_ok=True) 

    # Группируем по изображениям
    from collections import defaultdict
    detections_by_image = defaultdict(list)
    for det in all_detections:
        detections_by_image[det['image']].append(det)

    for img_name, detections in detections_by_image.items():
        txt_path = txt_dir / f"{Path(img_name).stem}.txt"
        with open(txt_path, 'w') as f:
            for det in detections:
                # YOLO формат: class_id center_x center_y width height (нормализованные 0-1)
                cls_id = 0  # если один класс
                cx_norm = det['center_x'] / det['image_width']
                cy_norm = det['center_y'] / det['image_height']
                w_norm = det['width'] / det['image_width']
                h_norm = det['height'] / det['image_height']
                f.write(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")


    print("=" * 60)
    print(" ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60 + "\n")

    print(f"✓ Всего обработано изображений: {len(results)}")
    print(f"✓ Всего обнаружено клеток: {total_detections}")

    if len(results) > 0:
        print(f"✓ Среднее клеток на изображение: {total_detections / len(results):.1f}")

    if all_detections:
        confidences = [d['confidence'] for d in all_detections]
        print(f"✓ Средняя уверенность: {sum(confidences) / len(confidences):.1%}")
        print(f"✓ Мин. уверенность: {min(confidences):.1%}")
        print(f"✓ Макс. уверенность: {max(confidences):.1%}")

    print(f"\n Результаты сохранены:")
    print(f"   • Изображения: {output_dir}/")
    print(f"   • JSON: {json_path}")
    print(f"   • CSV: {csv_path}")
    print(f"   • TXT (YOLO): {txt_dir}/")

    print("\n Результаты в каталоге  runs/detect/test_results/ ")
    print("=" * 60 + "\n")

    # ==========================================
    # ВЫВОД ТАБЛИЦЫ (первые 10 детекций)
    # ==========================================
    if all_detections:
        print(" Первые 10 детекций (таблица):\n")
        df_display = df[['image', 'object_id', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']].head(10)
        print(df_display.to_string(index=False))
        print("\n" + "=" * 60 + "\n")

    return all_detections


if __name__ == '__main__':
    detections = test_model()
