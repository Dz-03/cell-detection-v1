
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
    График 4 ключевых метрик для сравнения моделей
    """
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Key Metrics for Model Comparison', fontsize=16, fontweight='bold')

    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))

    # 1. mAP@0.5 (главная метрика детекции)
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

    # 2. F1-Score (баланс precision/recall)
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

    # 3. Loss (Train vs Val) - переобучение
    ax = axes[1, 0]
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        ax.plot(epochs, df['train/box_loss'], linewidth=2.5, color='#0984E3', label='Train Loss', marker='o', markersize=2)
        ax.plot(epochs, df['val/box_loss'], linewidth=2.5, color='#FD79A8', label='Val Loss', marker='s', markersize=2)

        # Gap (переобучение)
        final_gap = df['val/box_loss'].iloc[-1] - df['train/box_loss'].iloc[-1]
        gap_status = '✓' if final_gap < 0.1 else '⚠' if final_gap < 0.2 else '❌'
        ax.text(0.02, 0.98, f'Final Gap: {final_gap:.3f} {gap_status}',
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

    # Расчет ключевых метрик
    precision = df['metrics/precision(B)'].values if 'metrics/precision(B)' in df.columns else None
    recall = df['metrics/recall(B)'].values if 'metrics/recall(B)' in df.columns else None

    if precision is not None and recall is not None:
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_f1 = float(np.max(f1))
        best_f1_epoch = int(np.argmax(f1))
    else:
        best_f1 = 0.0
        best_f1_epoch = 0

    # Overfitting gap
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

        # КЛЮЧЕВЫЕ МЕТРИКИ ДЛЯ СРАВНЕНИЯ
        'metrics': {
            # 1. Главная метрика
            'mAP50': float(metrics.box.map50),
            'mAP50_95': float(metrics.box.map),

            # 2. Баланс
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': best_f1,

            # 3. Качество обучения
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'overfit_gap': overfit_gap,

            # 4. Эффективность
            'training_time_minutes': training_time,
            'epochs_trained': len(df),
            'best_epoch': int(df['metrics/mAP50(B)'].idxmax()) if 'metrics/mAP50(B)' in df.columns else 0,
        },

        # Дополнительная информация
        'convergence': {
            'best_f1_epoch': best_f1_epoch,
            'early_stopped': len(df) < config['epochs'],
        }
    }

    # Сохранение
    json_path = save_dir / 'comparison_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    return comparison_data


def print_comparison_summary(comparison_data):
    """
    Красивый вывод метрик для сравнения
    """
    print("\n" + "=" * 70)
    print("📊 КЛЮЧЕВЫЕ МЕТРИКИ ДЛЯ СРАВНЕНИЯ МОДЕЛЕЙ")
    print("=" * 70)

    m = comparison_data['metrics']

    print(f"\n🎯 КАЧЕСТВО ДЕТЕКЦИИ:")
    print(f"   mAP@0.5:      {m['mAP50']:.4f}  ⭐ (главная метрика)")
    print(f"   mAP@0.5:0.95: {m['mAP50_95']:.4f}")
    print(f"   F1-Score:     {m['f1_score']:.4f}  (баланс P/R)")

    print(f"\n⚖️  PRECISION & RECALL:")
    print(f"   Precision:    {m['precision']:.4f}")
    print(f"   Recall:       {m['recall']:.4f}")

    print(f"\n📉 КАЧЕСТВО ОБУЧЕНИЯ:")
    print(f"   Train Loss:   {m['final_train_loss']:.4f}")
    print(f"   Val Loss:     {m['final_val_loss']:.4f}")

    gap = m['overfit_gap']
    if gap < 0.1:
        status = "✓ Отлично"
        color = "🟢"
    elif gap < 0.2:
        status = "⚠ Приемлемо"
        color = "🟡"
    else:
        status = "❌ Переобучение"
        color = "🔴"
    print(f"   Overfit Gap:  {gap:.4f}  {color} {status}")

    print(f"\n⚡ ЭФФЕКТИВНОСТЬ:")
    print(f"   Время:        {m['training_time_minutes']:.1f} мин")
    print(f"   Эпох:         {m['epochs_trained']}")
    print(f"   Best Epoch:   {m['best_epoch']}")

    print("\n" + "=" * 70)


def compare_models(comparison_files):
    """
    Сравнение нескольких моделей
    """
    if len(comparison_files) < 2:
        print("⚠ Нужно минимум 2 модели для сравнения")
        return

    models_data = []
    for file_path in comparison_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            models_data.append(json.load(f))

    # Создание таблицы сравнения
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    headers = ['Model', 'mAP@0.5', 'F1-Score', 'Precision', 'Recall', 'Overfit Gap', 'Time (min)']
    table_data = []

    for i, data in enumerate(models_data):
        m = data['metrics']
        config_name = data['config'].get('name', f'Model_{i+1}')

        # Цветовая индикация для overfit gap
        gap = m['overfit_gap']
        gap_indicator = '✓' if gap < 0.1 else '⚠' if gap < 0.2 else '❌'

        table_data.append([
            config_name,
            f"{m['mAP50']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{gap:.4f} {gap_indicator}",
            f"{m['training_time_minutes']:.1f}"
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.15, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Стилизация заголовков
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Подсветка лучших значений
    metrics_to_highlight = ['mAP@0.5', 'F1-Score', 'Precision', 'Recall']
    for col_idx, header in enumerate(headers):
        if header in metrics_to_highlight:
            values = [float(row[col_idx].split()[0]) for row in table_data]
            best_idx = values.index(max(values))
            table[(best_idx + 1, col_idx)].set_facecolor('#FFD700')
            table[(best_idx + 1, col_idx)].set_text_props(weight='bold')

    plt.title('🏆 Models Comparison Table', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    return fig


def main():
    torch.cuda.empty_cache()

    data_yaml = Path('cell-detection.yolov8/data.yaml')
    if not data_yaml.exists():
        print(f" Файл {data_yaml} не найден!")
        return

    print("\n" + "=" * 70)
    print("  ОБУЧЕНИЕ YOLOV8 - ДЕТЕКЦИЯ КЛЕТОК")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("\n⚠ GPU не доступна, используется CPU")

    # Конфигурация для сохранения
    config = {
        'name': 'yolov8n_baseline',  # ← МЕНЯЙТЕ ДЛЯ РАЗНЫХ ЭКСПЕРИМЕНТОВ
        'model': 'yolov8n',
        'epochs': 100,
        'batch': 8,
        'imgsz': 1280,
        'augmentations': {
            'degrees': 15.0,
            'translate': 0.1,
            'scale': 0.3,
            'mosaic': 1.0,
            'mixup': 0.1,
        }
    }

    print(f"\n📝 Конфигурация: {config['name']}")

    model = YOLO('yolov8n.pt')

    print("\n⏳ Запуск обучения...")
    start_time = datetime.now()

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=0,

            # Аугментации
            degrees=config['augmentations']['degrees'],
            translate=config['augmentations']['translate'],
            scale=config['augmentations']['scale'],
            flipud=0.5,
            fliplr=0.5,
            mosaic=config['augmentations']['mosaic'],
            mixup=config['augmentations']['mixup'],
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
            name=config['name'],  # Используем имя из конфига
            exist_ok=True,
            verbose=True,
        )

        training_time = (datetime.now() - start_time).total_seconds() / 60

        print("\n✓ ОБУЧЕНИЕ ЗАВЕРШЕНО!")

        # save_dir = Path('runs/train') / config['name']
        save_dir = Path("runs/detect/runs/train/yolov8n_baseline")
        # Валидация
        print("\n⏳ Валидация...")
        best_model = YOLO(save_dir / 'weights' / 'best.pt')
        metrics = best_model.val()



        # ==========================================
        # СОХРАНЕНИЕ МЕТРИК ДЛЯ СРАВНЕНИЯ
        # ==========================================
        comparison_data = save_comparison_metrics(save_dir, metrics, training_time, config)
        print(f"\n✓ Метрики сохранены: {save_dir / 'comparison_metrics.json'}")

        # Вывод сводки
        print_comparison_summary(comparison_data)

        # ==========================================
        # ПОСТРОЕНИЕ ГРАФИКОВ
        # ==========================================
        print("\n⏳ Построение графиков...")

        results_csv = save_dir / 'results.csv'
        if results_csv.exists():
            fig = plot_key_metrics(results_csv)
            fig_path = save_dir / 'key_metrics.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ График сохранен: {fig_path}")
            plt.show()

        print("\n" + "=" * 70)
        print("✓ Готово! Метрики сохранены для сравнения")
        print("=" * 70)

        print(f"\n Для сравнения моделей запустите:")
        print(f"   comparison_files = [")
        print(f"       'runs/train/model1/comparison_metrics.json',")
        print(f"       'runs/train/model2/comparison_metrics.json',")
        print(f"   ]")
        print(f"   fig = compare_models(comparison_files)")
        print(f"   plt.show()")

    except Exception as e:
        print(f"\n Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()