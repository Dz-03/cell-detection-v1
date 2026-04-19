import os
import shutil
from pathlib import Path
import random


def split_dataset_exact(dataset_path, train_count=46, valid_count=5, test_count=2):
    """
    Разделяет train на train/valid/test с точными количествами

    Args:
        dataset_path: путь к датасету
        train_count: количество для train (46)
        valid_count: количество для valid (5)
        test_count: количество для test (2)
    """

    dataset_path = Path(dataset_path)
    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'

    # Проверка
    if not train_images_dir.exists():
        print(f"❌ Папка {train_images_dir} не найдена!")
        return

    # Получаем все изображения
    image_files = sorted(list(train_images_dir.glob('*.jpg')) +
                         list(train_images_dir.glob('*.png')) +
                         list(train_images_dir.glob('*.jpeg')))

    total_files = len(image_files)
    expected_total = train_count + valid_count + test_count

    print(f"📊 Найдено файлов: {total_files}")
    print(f"📊 Ожидается: {expected_total} (train:{train_count} + valid:{valid_count} + test:{test_count})")

    if total_files != expected_total:
        print(f"\n⚠️  ВНИМАНИЕ: Количество файлов не совпадает!")
        print(f"   Найдено: {total_files}, ожидается: {expected_total}")

        response = input(f"\nПродолжить с {total_files} файлами? (y/n): ")
        if response.lower() != 'y':
            return

        # Пересчитываем пропорции
        ratio = total_files / expected_total
        train_count = int(train_count * ratio)
        valid_count = int(valid_count * ratio)
        test_count = total_files - train_count - valid_count

        print(f"\n📊 Новое разделение:")
        print(f"   Train: {train_count}")
        print(f"   Valid: {valid_count}")
        print(f"   Test:  {test_count}")

    # Перемешиваем для случайного разделения
    random.seed(42)
    random.shuffle(image_files)

    # Разделяем
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:train_count + valid_count]
    test_files = image_files[train_count + valid_count:]

    print(f"\n📂 Разделение:")
    print(f"   🟢 Train: {len(train_files)} файлов")
    print(f"   🟡 Valid: {len(valid_files)} файлов")
    print(f"   🔵 Test:  {len(test_files)} файлов")

    # Создаём папки для valid и test
    for split in ['valid', 'test']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Функция перемещения файлов
    def move_files(file_list, target_split):
        moved_img = 0
        moved_lbl = 0
        missing_lbl = []

        for img_path in file_list:
            # Изображение
            target_img = dataset_path / target_split / 'images' / img_path.name
            shutil.move(str(img_path), str(target_img))
            moved_img += 1

            # Аннотация
            label_name = img_path.stem + '.txt'
            label_path = train_labels_dir / label_name

            if label_path.exists():
                target_lbl = dataset_path / target_split / 'labels' / label_name
                shutil.move(str(label_path), str(target_lbl))
                moved_lbl += 1
            else:
                missing_lbl.append(img_path.name)

        return moved_img, moved_lbl, missing_lbl

    # Перемещаем файлы
    print(f"\n📦 Перемещение файлов...")

    valid_img, valid_lbl, valid_missing = move_files(valid_files, 'valid')
    print(f"   🟡 Valid: {valid_img} изображений, {valid_lbl} аннотаций")
    if valid_missing:
        print(f"      ⚠️  Отсутствуют аннотации для: {', '.join(valid_missing[:3])}")

    test_img, test_lbl, test_missing = move_files(test_files, 'test')
    print(f"   🔵 Test:  {test_img} изображений,{test_lbl} аннотаций ")
    if test_missing:
        print(f"      ⚠️  Отсутствуют аннотации для: {', '.join(test_missing[:3])}")

    # Итоговая статистика
    print(f"\n" + "=" * 60)
    print(f"✅ Разделение завершено!")
    print(f"\n📁 Финальная структура:\n")

    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / 'images'
    lbl_dir = dataset_path / split / 'labels'

    img_count = len(list(img_dir.glob('*.[jp][pn]g'))) + len(list(img_dir.glob('*.jpeg')))
    lbl_count = len(list(lbl_dir.glob('*.txt')))

    status = "✅" if img_count == lbl_count else "⚠️"

    print(f"{status} {split}/")
    print(f"   images: {img_count} файлов")
    print(f"   labels: {lbl_count} файлов")
    print()

    # Проверяем data.yaml
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        print(f"📄 Проверьте data.yaml:")
    with open(data_yaml, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"\n{content}\n")

    # Проверяем, правильно ли указаны пути
    if 'val: valid/images' in content or 'val: valid\\images' in content:
        print(f"✅ data.yaml настроен правильно!")
    else:
        print(f"⚠️  data.yaml может требовать исправления!")
    print(f"\n📝 Должно быть:")
    print(f"""
path: {dataset_path.absolute()}
train: train/images
val: valid/images
test: test/images

names:
  0: cell

nc: 1
""")

    # Использование
    split_dataset_exact(
        dataset_path='../cell-detection.yolov8',  # Замените на имя вашей папки
        train_count=46,
        valid_count=5,
        test_count=2
    )