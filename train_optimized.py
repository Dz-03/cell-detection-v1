from ultralytics import YOLO
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from datetime import datetime


def plot_key_metrics(results_csv):
    """
    График 4 ключевых метрик
    """
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Key Metrics for Model Comparison', fontsize=16, fontweight='bold')

    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))

    # 1. mAP@0.5
    ax = axes[0, 0]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50(B)'], linewidth=2.5, color='#2E86DE', marker='o', markersize=3)
        best_val = df['metrics/mAP50(B)'].max()
        best_epoch = df['metrics/mAP50(B)'].idxmax()
        ax.axhline(y=best_val, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_val:.3f}')
        ax.scatter(best_epoch, best_val, color='red', s=200, zorder=5, marker='*')
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax.set_ylabel('mAP@0.5', fontweight='bold', fontsize=11)
    ax.set_title('mAP@0.5 (Main Metric)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 2. F1-Score
    ax = axes[0, 1]
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        precision = df['metrics/precision(B)'].values
        recall = df['metrics/recall(B)'].values
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        ax.plot(epochs, f1, linewidth=2.5, color='#10AC84', marker='s', markersize=3)
        best_f1 = np.max(f1)
        best_f1_epoch = np.argmax(f1)
        ax.axhline(y=best_f1, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_f1:.3f}')
        ax.scatter(best_f1_epoch, best_f1, color='red', s=200, zorder=5, marker='*')
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax.set_ylabel('F1-Score', fontweight='bold', fontsize=11)
    ax.set_title('F1-Score (Balance)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 3. Loss
    ax = axes[1, 0]
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        ax.plot(epochs, df['train/box_loss'], linewidth=2.5, color='#0984E3', label='Train Loss', marker='o',
                markersize=2)
        ax.plot(epochs, df['val/box_loss'], linewidth=2.5, color='#FD79A8', label='Val Loss', marker='s', markersize=2)

        final_gap = df['val/box_loss'].iloc[-1] - df['train/box_loss'].iloc[-1]
        gap_status = 'OK' if final_gap < 0.1 else 'Warning' if final_gap < 0.2 else 'Overfit'
        ax.text(0.02, 0.98, f'Final Gap: {final_gap:.3f} ({gap_status})',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax.set_title('Loss (Overfitting Check)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Precision & Recall
    ax = axes[1, 1]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(epochs, df['metrics/precision(B)'], linewidth=2.5, color='#6C5CE7',
                label='Precision', marker='o', markersize=3)
    if 'metrics/recall(B)' in df.columns:
        ax.plot(epochs, df['metrics/recall(B)'], linewidth=2.5, color='#FDCB6E',
                label='Recall', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax.set_title('Precision & Recall', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


def save_comparison_metrics(save_dir, metrics, training_time, config):
    """
    Сохранение ключевых метрик для сравнения в JSON
    """
    results_csv = save_dir / 'results.csv'
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    precision = df['metrics/precision(B)'].values if 'metrics/precision(B)' in df.columns else None
    recall = df['metrics/recall(B)'].values if 'metrics/recall(B)' in df.columns else None

    if precision is not None and recall is not None:
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_f1 = float(np.max(f1))
        best_f1_epoch = int(np.argmax(f1))
    else:
        best_f1 = 0.0
        best_f1_epoch = 0

    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        final_train_loss = float(df['train/box_loss'].iloc[-1])
        final_val_loss = float(df['val/box_loss'].iloc[-1])
        overfit_gap = final_val_loss - final_train_loss
    else:
        final_train_loss = 0.0
        final_val_loss = 0.0
        overfit_gap = 0.0

    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'config': config,

        'metrics': {
            'mAP50': float(metrics.box.map50),
            'mAP50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': best_f1,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'overfit_gap': overfit_gap,
            'training_time_minutes': training_time,
            'epochs_trained': len(df),
            'best_epoch': int(df['metrics/mAP50(B)'].idxmax()) if 'metrics/mAP50(B)' in df.columns else 0,
        },

        'convergence': {
            'best_f1_epoch': best_f1_epoch,
            'early_stopped': len(df) < config['epochs'],
        }
    }

    json_path = save_dir / 'comparison_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    return comparison_data


def print_comparison_summary(comparison_data):
    """
    Красивый вывод метрик для сравнения
    """
    print("\n" + "=" * 70)
    print("КЛЮЧЕВЫЕ МЕТРИКИ ДЛЯ СРАВНЕНИЯ МОДЕЛЕЙ")
    print("=" * 70)

    m = comparison_data['metrics']

    print(f"\nКАЧЕСТВО ДЕТЕКЦИИ:")
    print(f"   mAP@0.5:      {m['mAP50']:.4f}  (главная метрика)")
    print(f"   mAP@0.5:0.95: {m['mAP50_95']:.4f}")
    print(f"   F1-Score:     {m['f1_score']:.4f}  (баланс P/R)")

    print(f"\nPRECISION & RECALL:")
    print(f"   Precision:    {m['precision']:.4f}")
    print(f"   Recall:       {m['recall']:.4f}")

    print(f"\nКАЧЕСТВО ОБУЧЕНИЯ:")
    print(f"   Train Loss:   {m['final_train_loss']:.4f}")
    print(f"   Val Loss:     {m['final_val_loss']:.4f}")

    gap = m['overfit_gap']
    if gap < 0.1:
        status = "Отлично"
        color = "OK"
    elif gap < 0.2:
        status = "Приемлемо"
        color = "Warning"
    else:
        status = "Переобучение"
        color = "Alert"
    print(f"   Overfit Gap:  {gap:.4f}  [{color}] {status}")

    print(f"\nЭФФЕКТИВНОСТЬ:")
    print(f"   Время:        {m['training_time_minutes']:.1f} мин")
    print(f"   Эпох:         {m['epochs_trained']}")
    print(f"   Best Epoch:   {m['best_epoch']}")

    print("\n" + "=" * 70)


def main():
    torch.cuda.empty_cache()

    data_yaml = Path('cell-detection.yolov8/data.yaml')
    if not data_yaml.exists():
        print(f"Файл {data_yaml} не найден!")
        return

    print("\n" + "=" * 70)
    print("  ОПТИМИЗИРОВАННАЯ КОНФИГУРАЦИЯ - RTX 3060 LAPTOP (6GB)")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"VRAM: {vram_gb:.1f} GB")

        torch.backends.cudnn.benchmark = True
    else:
        print("\nGPU не доступна!")
        return

    # ==========================================
    # ОПТИМИЗИРОВАННАЯ КОНФИГУРАЦИЯ ДЛЯ 6GB VRAM
    # ==========================================
    config = {
        'name': 'yolov8m_optimized_6gb',
        'model': 'yolov8m',  # Medium (25M параметров) вместо X (68M)
        'epochs': 200,  # Уменьшили с 300
        'batch': 8,  #  16 не поместится
        'imgsz': 960,  # Уменьшили с 1280 (оптимальный баланс)

        # Аугментации (немного урезанные)
        'augmentations': {
            'degrees': 20.0,
            'translate': 0.2,
            'scale': 0.8,  # Уменьшили с 0.9
            'shear': 5.0,
            'perspective': 0.0003,  # Уменьшили
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,  # Уменьшили с 0.15
            'copy_paste': 0.2,  # Уменьшили с 0.3
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'erasing': 0.3,  # Уменьшили с 0.4
        },

        # Оптимизация
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,  # Уменьшили с 5
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,

        # Multi-scale
        'close_mosaic': 15,  # Уменьшили с 20
    }

    print(f"\nКонфигурация: {config['name']}")
    print(f"Модель: {config['model']} (25M параметров)")
    print(f"Разрешение: {config['imgsz']}x{config['imgsz']}")
    print(f"Batch size: {config['batch']}")
    print(f"Эпох: {config['epochs']}")

    # Оценка использования VRAM
    estimated_vram = 1.2 + (config['batch'] * (config['imgsz'] / 640) ** 2 * 0.4)
    print(f"\nОжидаемое использование VRAM: ~{estimated_vram:.1f} GB")

    if estimated_vram > vram_gb * 0.9:
        print("   ПРЕДУПРЕЖДЕНИЕ: Возможна нехватка VRAM!")

    model = YOLO(f"{config['model']}.pt")

    print("\nЗапуск обучения...")
    print("Ожидаемое время: ~3-5 часов\n")

    start_time = datetime.now()

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=0,
            workers=6,  # Уменьшили с 8 (экономия RAM)

            # Оптимизатор
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],

            # Loss weights
            box=config['box'],
            cls=config['cls'],
            dfl=config['dfl'],

            # Аугментации
            degrees=config['augmentations']['degrees'],
            translate=config['augmentations']['translate'],
            scale=config['augmentations']['scale'],
            shear=config['augmentations']['shear'],
            perspective=config['augmentations']['perspective'],
            flipud=config['augmentations']['flipud'],
            fliplr=config['augmentations']['fliplr'],
            mosaic=config['augmentations']['mosaic'],
            mixup=config['augmentations']['mixup'],
            copy_paste=config['augmentations']['copy_paste'],
            hsv_h=config['augmentations']['hsv_h'],
            hsv_s=config['augmentations']['hsv_s'],
            hsv_v=config['augmentations']['hsv_v'],
            erasing=config['augmentations']['erasing'],

            # Дополнительные настройки
            amp=True,  # КРИТИЧНО для экономии VRAM!
            patience=40,  # Уменьшили с 50
            close_mosaic=config['close_mosaic'],

            # Сохранение
            save=True,
            save_period=20,  # Реже сохраняем (экономия места)
            plots=True,
            project='runs/train',
            name=config['name'],
            exist_ok=True,
            verbose=True,

            # Валидация
            val=True,
            fraction=1.0,

            # Кэширование (экономия времени)
            cache='ram',  # Кэшировать в RAM (если хватает)
        )

        training_time = (datetime.now() - start_time).total_seconds() / 60

        print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")

        save_dir = Path(model.trainer.save_dir)

        # Вариант 2: Если вариант 1 не работает, ищем последнюю папку
        if not save_dir.exists() or not (save_dir / 'weights' / 'best.pt').exists():
            print("\n⚠️  Автоопределение пути не сработало, ищем вручную...")
            base_dir = Path('runs/train')

            # Находим все папки с нужным именем
            matching_dirs = list(base_dir.glob(f"{config['name']}*"))

            if matching_dirs:
                # Берем самую свежую
                save_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
                print(f"   Найдена папка: {save_dir}")
            else:
                print(f" Папка с результатами не найдена!")
                print(f"   Ожидалась: runs/train/{config['name']}")
                print(f"   Проверьте содержимое runs/train/")
                return

        print(f"\n Папка с результатами: {save_dir}")

        # Проверяем наличие best.pt
        best_model_path = save_dir / 'weights' / 'best.pt'
        if not best_model_path.exists():
            print(f"\n Файл best.pt не найден: {best_model_path}")
            print(f"   Содержимое {save_dir / 'weights'}:")
            if (save_dir / 'weights').exists():
                for f in (save_dir / 'weights').iterdir():
                    print(f"      - {f.name}")
            return

        # Валидация
        print("\n Валидация лучшей модели...")
        best_model = YOLO(best_model_path)
        metrics = best_model.val(
            data=str(data_yaml),
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=0,
            plots=True,
            save_json=True,
        )

        # Сохранение метрик
        comparison_data = save_comparison_metrics(save_dir, metrics, training_time, config)
        print(f"\n Метрики сохранены: {save_dir / 'comparison_metrics.json'}")

        print_comparison_summary(comparison_data)

        # Построение графиков
        print("\n Построение графиков...")
        results_csv = save_dir / 'results.csv'
        if results_csv.exists():
            fig = plot_key_metrics(results_csv)
            fig_path = save_dir / 'key_metrics.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f" График сохранен: {fig_path}")
            plt.show()
        else:
            print(f"  Файл results.csv не найден: {results_csv}")

        print("\n" + "=" * 70)
        print(" ГОТОВО! Оптимизированная модель обучена")
        print("=" * 70)

        print(f"\n Ожидаемые результаты:")
        print(f"   mAP@0.5:      79.9% → 85-89%")
        print(f"   mAP@0.5:0.95: 40.6% → 52-60%")
        print(f"   Recall:       71.1% → 80-85%")
        print(f"   F1-Score:     74.5% → 83-87%")

        print(f"\n Файлы модели:")
        print(f"   Best:  {save_dir / 'weights' / 'best.pt'}")
        print(f"   Last:  {save_dir / 'weights' / 'last.pt'}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n ОШИБКА: Недостаточно VRAM!")
            torch.cuda.empty_cache()
        else:
            raise e
    except Exception as e:
        print(f"\n Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
