from ultralytics import YOLO
from pathlib import Path
import glob


def test_model():
    print("\n" + "=" * 60)
    print("ТЕСТ ДЕТЕКЦИИ КЛЕТОК")
    print("=" * 60 + "\n")

    # Находим модель
    models = glob.glob('models/yolov8m_optimized_6gb/weights/best.pt', recursive=True)
    if not models:
        print(" Модель не найдена!")
        return

    model_path = max(models, key=lambda p: Path(p).stat().st_mtime)
    print(f" Модель: {model_path}\n")

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
    print(" Запуск детекции...\n")

    results = model.predict(
        source=test_images[:10],
        save=True,
        conf=0.25,  # Порог уверенности
        iou=0.45,
        device='cpu',
        project='runs/detect',
        name='test_results',
    )

    # Статистика
    print("=" * 60)
    print(" РЕЗУЛЬТАТЫ ДЕТЕКЦИИ")
    print("=" * 60 + "\n")

    total_detections = 0
    for i, result in enumerate(results):
        num_boxes = len(result.boxes)
        total_detections += num_boxes
        print(f"   Изображение {i + 1}: {num_boxes} клеток обнаружено")

    print(f"\n Всего обнаружено клеток: {total_detections}")
    print(f" Результаты сохранены в: runs/detect/test_results/")

    print("\n Откройте папку runs/detect/test_results/ чтобы посмотреть результаты!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_model()