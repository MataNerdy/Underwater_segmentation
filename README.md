# Underwater Segmentation

Портфолио-проект по сегментации подводных изображений на PyTorch. Репозиторий
заменяет notebook-формат на воспроизводимый код с отдельными модулями для
датасета, метрик, обучения, оценки, инференса и визуализации.

## STAR

### Situation

Подводные изображения часто имеют низкий контраст, цветовые искажения и шум.
Для таких данных важно не только обучить модель сегментации, но и показать
аккуратный инженерный пайплайн: подготовка данных, обучение, оценка качества,
сохранение предсказаний и понятная структура проекта.

### Task

Цель проекта — собрать чистый PyTorch-репозиторий, который можно показать в
портфолио и быстро запустить на локальной машине или в облачной среде без
жестко заданных путей к Google Drive и без абсолютных путей.

### Action

- Логика датасета вынесена в `src/dataset.py`.
- Метрики Dice, IoU и pixel accuracy вынесены в `src/metrics.py`.
- Обучение доступно через CLI в `src/train.py`.
- Оценка модели доступна через `src/evaluate.py`.
- Инференс и файл `submission.txt` создаются через `src/predict.py`.
- Визуальные примеры предсказаний сохраняются через `src/visualize.py`.
- Данные, чекпоинты и большие артефакты исключены из Git через `.gitignore`.

### Result

Проект стал воспроизводимым и удобным для ревью: код разделен по зонам
ответственности, команды запуска документированы, а артефакты обучения не
попадают в репозиторий.

## Структура

```text
src/
  dataset.py      # загрузка изображений и масок
  metrics.py      # Dice, IoU, pixel accuracy
  model.py        # компактная U-Net модель
  train.py        # обучение
  evaluate.py     # оценка
  predict.py      # предсказания и submission.txt
  visualize.py    # визуальные примеры
tests/
  test_smoke.py   # быстрый smoke test
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Формат данных

Ожидается парная структура директорий. Имена изображений и масок должны
совпадать по stem, например `001.png` и `001.png`.

```text
data/
  train_images/
    001.png
  train_masks/
    001.png
  test_images/
    101.png
```

## Обучение

```bash
python -m src.train \
  --images-dir data/train_images \
  --masks-dir data/train_masks \
  --output-dir checkpoints \
  --epochs 20 \
  --batch-size 8 \
  --image-size 256
```

## Оценка

```bash
python -m src.evaluate \
  --images-dir data/valid_images \
  --masks-dir data/valid_masks \
  --checkpoint checkpoints/best_model.pth
```

## Предсказания

```bash
python -m src.predict \
  --images-dir data/test_images \
  --checkpoint checkpoints/best_model.pth \
  --submission-path submission.txt
```

## Визуализация

```bash
python -m src.visualize \
  --images-dir data/test_images \
  --checkpoint checkpoints/best_model.pth \
  --output-dir outputs/visualizations
```

## Проверка

```bash
pytest
```

## Примечания

Репозиторий не должен хранить датасеты, чекпоинты, архивы и большие файлы.
Используйте относительные пути через CLI-аргументы.
