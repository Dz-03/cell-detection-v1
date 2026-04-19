from ultralytics import YOLO
from pathlib import Path
import glob


def main():
    # Автоматический поиск best.pt
    best_pt_files = glob.glob('runs/**/best.pt', recursive=True)

    if best_pt_files:
        best_path = Path(best_pt_files[0])
        save_dir = best_path.parent.parent
        print(f"✅ Найдена модель: {best_path}")
        print(f"   Папка результатов: {save_dir}")

        # Загружаем модель
        best_model = YOLO(best_path)

        # Валидация с отключенной многопоточностью
        metrics = best_model.val(
            workers=0,  # КЛЮЧЕВОЙ ПАРАМЕТР для Windows!
            batch=8,
            device=0,
            plots=True  # Сохранить графики
        )

        # Вывод результатов
        print(f"\n📊 Итоговые метрики:")
        print(f"   mAP50:     {metrics.box.map50:.3f}")
        print(f"   mAP50-95:  {metrics.box.map:.3f}")
        print(f"   Precision: {metrics.box.mp:.3f}")
        print(f"   Recall:    {metrics.box.mr:.3f}")

    else:
        print("❌ Модель best.pt не найдена!")


# ЗАЩИТА ДЛЯ WINDOWS - ОБЯЗАТЕЛЬНО!
if __name__ == '__main__':
    main()