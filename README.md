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
│   └── [coming soon] pruning/         # Эксперименты со спарсификацией
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

coming soon...

### 4. Измерение производительности
Файл: src/core/latency.py

Метрики: Время выполнения, использование GPU памяти

Платформы: CPU и GPU измерения с синхронизацией

### 5. Тестовая модель

coming soon...

## Быстрый старт
### Установка зависимостей
```bash
pip install torch torchvision numpy ...
```
### Базовое использование
```python
...
```
### Запуск экспериментов
```python
from src.core.latency import latency_gpu
from src.models.dummy import DummyModel
...
```
### Экспериментальная часть
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

<img width="544" height="460" alt="image" src="https://github.com/user-attachments/assets/0f26cb06-9f74-46e7-9f20-9f21d64bfa54" />

### Воспроизведение результатов

coming soon...

## Технические детали
### Требования к оборудованию
* GPU с поддержкой CUDA (рекомендуется)
* PyTorch 1.9+
* Python 3.7+
...

