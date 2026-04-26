"""
Обучение YOLOv8 для детекции клеток
"""

from ultralytics import YOLO
import torch
from pathlib import Path


def main():
    # Очистка GPU
    torch.cuda.empty_cache()

    # Проверка датасета
    data_yaml = Path('cell-detection.yolov8/data.yaml')
    if not data_yaml.exists():
        print(f"Файл {data_yaml} не найден!")
        print(f"Текущая директория: {Path.cwd()}")
        return

    print("\n" + "=" * 60)
    print("  ОБУЧЕНИЕ YOLOV8 - ДЕТЕКЦИЯ КЛЕТОК")
    print("=" * 60)

    # Информация о системе
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("\nGPU не доступна, используется CPU")

    print(f"Датасет: {data_yaml}")

    # Инициализация модели
    print("\n Загрузка YOLOv8n...")
    model = YOLO('yolov8n.pt')

    # Обучение
    print("\n Запуск обучения...")
    print("=" * 60 + "\n")

    try:
        results = model.train(
            data=str(data_yaml),

            # Основные параметры
            epochs=100,
            imgsz=640,
            batch=8,
            device=0 if torch.cuda.is_available() else 'cpu',

            # ВАЖНО для Windows!
            workers=0,  # Отключаем многопоточность

            # Аугментации
            degrees=15.0,
            translate=0.1,
            scale=0.3,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
           
            # Цветовые
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,

            # Оптимизация
            amp=True,
            patience=20,

            # Сохранение
            save=True,
            plots=True,
            project='runs/train',
            name='cell_detection',
            exist_ok=True,

            # Вывод
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 60)

        # Пути к результатам
        save_dir = Path('runs/train/cell_detection')
        print(f"\n Результаты:")
        print(f"   Папка:        {save_dir}")
        print(f"   Лучшая модель: {save_dir}/weights/best.pt")
        print(f"   Последняя:    {save_dir}/weights/last.pt")
        print(f"   Графики:      {save_dir}/results.png")

        # Валидация
        print("\n Валидация лучшей модели...")
        best_model = YOLO(save_dir / 'weights' / 'best.pt')
        metrics = best_model.val()

        print(f"\n Финальные метрики:")
        print(f"   mAP50:     {metrics.box.map50:.3f}")
        print(f"   mAP50-95:  {metrics.box.map:.3f}")
        print(f"   Precision: {metrics.box.mp:.3f}")
        print(f"   Recall:    {metrics.box.mr:.3f}")

        print("\n" + "=" * 60)
        print(" Всё готово! Модель обучена успешно!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
