# conv2d_reimagined
Effective Deep Learning Models - Conv2d Reimagined: img2col, GEMM, Sparsity &amp; Quantization

## Обзор проекта
Этот проект посвящён исследованию и оптимизации реализации свёрточных слоёв в нейронных сетях. Основной фокус - преобразование операции свёртки в эффективное матричное умножение с последующим применением современных методов оптимизации.

Мы реализовали кастомную реализацию свёрточного слоя Conv2D в PyTorch с использованием метода img2col — классического подхода к ускорению свёрток за счёт преобразования локальных патчей изображения в столбцы матрицы. Эта реализация выполнена в виде автономной PyTorch-функции с поддержкой автоматического дифференцирования, совместимой с обучением и квантованием (QAT — Quantization-Aware Training).

## Ключевые цели:
* Исследование классической реализации nn.Conv2d в PyTorch

* Реализация преобразования свёртки в форму img2col для сведения к матричному умножению (GEMM)

* Применение методов оптимизации: спарсификация весов и квантование

* Экспериментальная оценка производительности и использования памяти

## Структура проекта
```txt
conv2d_reimagined
├── experiments/
│   ├── common/
│   │   ├── conv2d_img2col.py          # Базовая реализация Img2Col + GEMM
│   │   └── replace_conv_resnet.py     # Утилиты замены свёрточных слоёв
│   ├── conv2d_img2col_QAT.py          # Квантованный Img2Col (QAT)
│   ├── quantize.ipynb                 # Эксперименты по квантизации
│   └── prune.py                       # Реализация классов прунинга
│   └── pruning_benchmark.ipynb        # Эксперименты со спарсификацией
├── setup/
│   └── setup-gpu.sh                   # Установка необходимых зависимостей
├── src/
│   ├── core/
│   │   └── latency.py                 # Утилиты измерения производительности
│   ├── models/
│   │   └── dummy.py                   # Тестовые модели для экспериментов
│   └── utils.py                       # Расширенные методы квантования
├── tests/
│   └── test_conv2d.py                 # Тест работы модифицированного сверточного слоя
└── README.md
```
## Реализованные компоненты
### 1. Img2Col + GEMM преобразование
Файл: experiments/common/conv2d_img2col.py

Классы: Img2ColConvFunction, Conv2dImg2Col

Особенности: Автоматическое дифференцирование, поддержка различных параметров свёртки

### 2. Квантование (QAT - Quantization Aware Training)
Файл: experiments/conv2d_img2col_QAT.py

Поддержка: FP16 → INT8 квантование весов и активаций

Интеграция: Совместимость с PyTorch FX Graph Mode Quantization

### 3. Спарсификация

Файл: experiments/prune.py

Классы: MagnitudePruner, IterativePruner

Особенности: Модификация итеративного прунинга, которая позволяет фиксировать sparsity на несколько эпох перед ее повышением

### 4. Измерение производительности
Файл: src/core/latency.py

Метрики: Время выполнения, использование GPU памяти

Платформы: CPU и GPU измерения с синхронизацией

## Быстрый старт
### Настройка окружения

В данном проекте в качестве окружения выступает devcontainer. Как его запустить, см. подробнее в документации:

https://code.visualstudio.com/docs/devcontainers/containers

### Экспериментальная часть - /experiments/quantize.ipynb

### 1. **Микробенчмарк: Простые свёрточные сети**
- Сравниваются две последовательные модели:
  - Блоки `Conv2dImg2Col` + ReLU
  - Стандартные блоки `nn.Conv2d` + ReLU
- Измеряется латентность при различных:
  - Размерах ядра: `[1, 3, 5, 7, 9, 11]`
  - Размерах батча: `[1, 2, 4, 8, 16, 32, 64]`
- Результаты сохраняются в `sequential_model_latency_dummy.csv`.

### 2. **Пайплайн квантования (QAT)**
- Применяется QAT с использованием `torch.ao.quantization`:
  - Полное квантование (все слои).
  - Частичное квантование (исключаются первые и последние слои, например `conv1`, `segmentation_head`).
- Используется `convert_fx` для финального преобразования в INT8.
- Поддержка моделей как со стандартными, так и с кастомными свёртками.

### 3. **Бенчмарк сегментации на CamVid**
Обучение и оценка модели `FPN + resnext50_32x4d` в различных конфигурациях:

| Точность | Режим обучения       | Тип свёртки           | Шаблон имени файла                     |
|----------|----------------------|------------------------|----------------------------------------|
| FP32     | Полное обучение      | Стандартная / Кастомная| `*_camvid_model_fp32.pt`               |
| FP16     | Смешанная точность   | Стандартная / Кастомная| `*_camvid_model_fp16.pt`               |
| TF32     | С включённым TF32    | Стандартная / Кастомная| `*_camvid_model_tf32.pt`               |
| INT8     | QAT + дообучение     | Стандартная / Кастомная| `*_int8-qat_x86*.pt`                   |

Для каждой модели:
- Измеряется **латентность** на CPU/GPU при разных размерах батча.
- Оценивается **точность** через mean IoU на валидационном и тестовом наборах.
- Результаты записываются через `LatencyMetricsWriter` в CSV-файлы.

### 4. **Экспорт моделей и готовность к развёртыванию**
- Экспорт квантованных моделей с помощью:
  - `torch.jit.trace`
  - `torch.jit.script`
  - (Подразумевается) экспорт в ONNX через `export_onnx`
- Обеспечивается совместимость с инференс-движками.

### 5. **Визуализация**
- Качественные результаты отображаются с помощью `visualize_sample` на тестовых изображениях CamVid.
- Данные бенчмарка анализируются функцией `analyze_benchmark_data`.

#### План экспериментов:

* Сравнение производительности: Forward/backward время для разных размеров ядер (3×3, 5×5, 7×7)

* Анализ памяти: Использование GPU памяти для различных batch sizes

* Оптимизации: Оценка эффекта от спарсификации и квантования

* Интеграция: Тестирование в реальной модели

* Прунинг. Методы: Structured pruning
* Квантование. Методы: ... 

#### Результаты экспериментов с исходным Conv2D слоем:

<img width="1280" height="640" alt="image" src="https://github.com/user-attachments/assets/4cbf431c-f1d5-4e42-88e9-cda9da73fcc8" />

int8:


<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/2f807dbf-7249-47f1-87a6-eb8123aad128" /> 

fp32:


<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/316b4102-ed40-4aa6-8256-5d52fbe66738" />

fp16:

<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/a127234d-a442-4e9d-9963-c7ea305194c9" />

fp16 half:

<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/7ed617fc-f911-40f0-8969-d872878653e7" />

int8 custom cfg:

<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/0f26cb06-9f74-46e7-9f20-9f21d64bfa54" />

#### Результаты экспериментов Conv2dImg2Col:

<img width="1280" height="640" alt="image" src="https://github.com/user-attachments/assets/4cbf431c-f1d5-4e42-88e9-cda9da73fcc8" />

int8:


<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/b0049aa9-7442-4f20-a89c-a1564846d72b" />


fp32:


<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/a0d1351f-b987-4bd6-88e5-a48aa3c8f02f" />


fp16:


<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/9f8ec3e0-3009-4541-a3b1-0cc956d6e799" />


fp16 half:

<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/91a728f1-aa9f-4565-a01d-25c3230fb10d" />


int8 custom cfg:

<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/d5ab23dd-8003-41df-a8d2-f413755bce4a" />

#### CPU Latency vs Batch Size:

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/458c1092-403d-4820-b908-0113e48239c7" />

#### GPU Latency vs Batch Size:

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/1d93dc58-3b8b-4ea5-bbda-15138bea7c1e" />

#### Test Accuracy vs Latency (CPU latency):

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/3bc3e44b-c1f6-483d-b5d5-3047255528b3" />

#### Test Accuracy vs Latency (GPU latency):

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/77d55853-e958-471d-a8af-12f432be935f" />

### Экспериментальная часть - /experiments/pruning_benchmark.ipynb

Пайплайн в большинстве своем случае дублирует пайплайн квантизиции за исключением наличия файн тюнинга моделей и логики итеративного прунинга

#### Результаты экспериментов с исходным Conv2D слоем:

magnitude_unstructured_sp03

<img width="1372" height="353" alt="image" src="https://github.com/user-attachments/assets/74513a9b-7195-4bd4-be9b-f98eb94ebf94" />

ReplacedConv_magnitude_unstructured_sp03

<img width="1371" height="349" alt="image" src="https://github.com/user-attachments/assets/84bf966c-6fd3-48b7-8946-3563c2d29d71" />

magnitude_structured_sp03

<img width="1370" height="356" alt="image" src="https://github.com/user-attachments/assets/115f8cdd-1923-4775-bfe0-ccc9af3bc547" />

ReplacedConv_magnitude_structured_sp03

<img width="1370" height="356" alt="image" src="https://github.com/user-attachments/assets/f44a41dc-c011-4592-a71b-664481cc443f" />

magnitude_unstructured_sp05

<img width="1370" height="356" alt="image" src="https://github.com/user-attachments/assets/39fe9a10-66e2-4c63-96f6-dcd6da93266c" />

ReplacedConv_magnitude_unstructured_sp05

<img width="1370" height="356" alt="image" src="https://github.com/user-attachments/assets/9e6a1236-7c43-437b-b0ba-d76d11e07ca8" />

magnitude_structured_sp05

<img width="1370" height="356" alt="image" src="https://github.com/user-attachments/assets/4fedadd7-a9fc-4c8d-b375-8b515fa1259b" />

ReplacedConv_magnitude_structured_sp05

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/50cf4877-22ad-4647-8a03-07476b222177" />

magnitude_unstructured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/afbd2fe1-7f5b-4908-a970-f3d1e27e9b47" />

ReplacedConv_magnitude_unstructured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/39ac2615-e64a-429a-b7e7-139e7e0989d9" />

magnitude_structured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/0e95f703-b5fe-4736-8304-c313c866dfb2" />

ReplacedConv_magnitude_structured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/315f8e88-19cb-428d-954d-fc996ba9f491" />

iterative_unstructured_sp05

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/a2eecde0-f6e9-4825-826e-905b7f9b7293" />

ReplacedConv_iterative_unstructured_sp05

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/fdc6e92e-d6af-403e-a876-cc05f2befa29" />

iterative_structured_sp05

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/3ccd085a-38a9-4d70-8921-8e28206303a8" />

ReplacedConv_iterative_structured_sp05

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/61dd3539-843f-4545-a02d-16ef6dc380cd" />

iterative_unstructured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/f10bd57e-a1b3-4f74-9ac0-84fe0eafe6d2" />

ReplacedConv_iterative_unstructured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/b7500994-3577-447e-9111-7813dec6eb17" />

iterative_structured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/92031113-8cc4-4106-8c2e-adbd2cbed6a5" />

ReplacedConv_iterative_structured_sp07

<img width="1369" height="346" alt="image" src="https://github.com/user-attachments/assets/c0119615-2ca8-4768-b12a-fb519505065d" />

#### Результаты экспериментов Conv2dImg2Col:

#### CPU Latency vs Batch Size:

<img width="816" height="514" alt="image" src="https://github.com/user-attachments/assets/a90a7a0e-fd7a-46ef-b23b-29021f055c86" />

#### GPU Latency vs Batch Size:

<img width="797" height="507" alt="image" src="https://github.com/user-attachments/assets/33a16917-587c-4014-bedb-f77e9dc5adc0" />

#### Test Accuracy vs Latency (CPU latency):

<img width="1096" height="441" alt="image" src="https://github.com/user-attachments/assets/8e1b5a17-7946-4c16-bf31-1fb70535e446" />


#### Test Accuracy vs Latency (GPU latency):

<img width="1100" height="437" alt="image" src="https://github.com/user-attachments/assets/cc0e6fb0-97e2-4b7f-99c3-5a6008e3516f" />


### Воспроизведение результатов

См. раздел с настройкой окружения и экспериментальной частью (файл /experiments/quantize.ipynb и /experiments/pruning_benchmark.ipynb (при наличии базовых моделей без квантизации))

### Веса моделей

https://drive.google.com/drive/folders/18cX7nAghs9GnRVO5o2__KVg2bkPUrkIS?usp=sharing
https://drive.google.com/drive/folders/1jHbeRjwmm2kwjRq2r2O9zGTjXrTe1Siv?usp=sharing

## Технические детали
### Требования к оборудованию
* GPU с поддержкой CUDA (рекомендуется)
* PyTorch 1.9+
* Python 3.7+
...

