# Underwater Segmentation

Портфолио-проект по 8-классовой семантической сегментации подводных сцен на
PyTorch. Проект поддерживает два baseline-подхода: U-Net и DeepLabV3 ResNet50.
Код вынесен из notebook-формата в воспроизводимые CLI-скрипты без Google Drive
и абсолютных локальных путей.


### Situation


Подводные изображения сложны для сегментации: в них часто есть шум, слабый
контраст и нестабильная цветопередача. Исходный baseline находился в notebook,
где смешивались загрузка данных, обучение, оценка, инференс и визуализации.

### Task

Нужно было превратить baseline в чистый репозиторий для портфолио: разделить
код по модулям, оставить конфигурируемые пути, сохранить U-Net baseline,
добавить DeepLabV3 ResNet50 как второй вариант модели и подготовить команды для
обучения, оценки, предсказаний и визуализации.

### Action

- `src/dataset.py` загружает изображения и RGB-маски, преобразуя маски в классы
  `0..7` по правилу `4 * R + 2 * G + B` после порога `> 100`.
- `src/metrics.py` считает pixel accuracy, mean IoU и IoU по каждому классу.
- `src/model.py` содержит компактный U-Net baseline и DeepLabV3 ResNet50 с
  выходом на 8 классов.
- `src/train.py` обучает выбранную модель через `CrossEntropyLoss`, поддерживает
  weighted `CrossEntropyLoss`, разные размеры изображений и запись результатов.
- `src/evaluate.py` оценивает чекпоинт на размеченных данных.
- `src/predict.py` сохраняет `submission.txt` и поддерживает horizontal flip TTA.
- `src/visualize.py` сохраняет примеры предсказаний в `assets/prediction_examples.png`.
- `.gitignore` исключает датасеты, чекпоинты, архивы и большие артефакты.

### Result

Репозиторий стал пригоден для ревью: notebook можно оставить как историю
эксперимента, а основной пайплайн запускается через CLI. Код отделяет датасет,
метрики, обучение, оценку и инференс, поэтому проект проще воспроизводить,
расширять и демонстрировать.

## Данные

Ожидаемая структура:

```text
underwater_data/
├── train/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

Имена изображений и масок должны совпадать по stem: например, `d_r_162_.jpg`
и `d_r_162_.bmp`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Обучение

### U-Net baseline

```bash
python src/train.py \
  --model unet \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4 \
  --image-size 256
```

### DeepLabV3 ResNet50 baseline

```bash
python src/train.py \
  --model deeplabv3_resnet50 \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4 \
  --image-size 256
```

Для инициализации pretrained backbone:

```bash
python src/train.py \
  --model deeplabv3_resnet50 \
  --pretrained-backbone \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4 \
  --image-size 256
```

Fine-tuning с меньшим learning rate:

```bash
python src/train.py \
  --model unet \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-5 \
  --image-size 256 \
  --experiment-name unet_lr1e-5
```

Weighted loss:

```bash
python src/train.py \
  --weighted-loss \
  --experiment-name unet_weighted_ce
```

Сравнение размеров изображений выполняется через `--image-size 128` и
`--image-size 256`. Итоги экспериментов добавляются в
`experiments/results.csv`, включая best validation pixel accuracy, mean IoU и
per-class IoU. Графики сохраняются в `assets/`.

План всех экспериментов с готовыми командами находится в
`experiments/experiment_plan.md`.

## Kaggle

Для Kaggle есть два удобных варианта запуска:

- notebook: `notebooks/kaggle_underwater_experiments.ipynb`;
- shell runner: `scripts/run_kaggle_experiments.sh`.

Runner по умолчанию запускает основные эксперименты на 20 эпох, потому что
лучший предыдущий результат был получен после 20 эпох:

```bash
chmod +x scripts/run_kaggle_experiments.sh
DATASET_NAME=your-kaggle-dataset-name ./scripts/run_kaggle_experiments.sh
```

По умолчанию путь к данным строится так:

```text
/kaggle/input/${DATASET_NAME}/underwater_data
```

Если структура датасета другая, задайте путь напрямую:

```bash
DATA_DIR=/kaggle/input/your-kaggle-dataset-name/underwater_data \
  ./scripts/run_kaggle_experiments.sh
```

Быстрый smoke-test режим запускает те же эксперименты на 1 эпоху:

```bash
DATASET_NAME=your-kaggle-dataset-name ./scripts/run_kaggle_experiments.sh --smoke
```

Runner выполняет:

- U-Net, `image_size=128`, `epochs=20`;
- U-Net, `image_size=256`, `epochs=20`;
- U-Net weighted CrossEntropyLoss, `image_size=256`, `epochs=20`;
- DeepLabV3 ResNet50, `image_size=256`, `epochs=20`;
- DeepLabV3 ResNet50 с pretrained backbone, `image_size=256`, `epochs=20`.

Все outputs сохраняются в `/kaggle/working`:

- checkpoints: `/kaggle/working/checkpoints`;
- results CSV: `/kaggle/working/experiments/results.csv`;
- plots: `/kaggle/working/assets`.

## Оценка

```bash
python src/evaluate.py --checkpoint checkpoints/best.pth
```

Скрипт печатает pixel accuracy, mean IoU и IoU для каждого класса.

## Предсказания

```bash
python src/predict.py \
  --checkpoint checkpoints/best.pth \
  --output submission.txt \
  --tta \
  --mode-filter
```

Файл `submission.txt` сохраняется в формате notebook baseline: заголовок с
формой массива и отдельные срезы масок классов. Опции `--tta` и
`--mode-filter` повторяют inference-улучшения из notebook: horizontal flip TTA
и сглаживание 3x3 majority vote.

## Визуализация

```bash
python src/visualize.py \
  --checkpoint checkpoints/best.pth \
  --output assets/prediction_examples.png
```

## Smoke Test

```bash
pytest
```


Тест проверяет кодирование RGB-масок, датасет, метрики и формат
`submission.txt` на синтетических данных.

## Что не хранится в Git

Датасеты, чекпоинты, `submission.txt`, архивы, generated `experiments/*.csv`,
`assets/` и другие генерируемые файлы исключены из репозитория. План
экспериментов `experiments/experiment_plan.md` хранится в Git. Все пути
передаются через CLI-аргументы.
