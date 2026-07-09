# GPU и облачная инфраструктура

> Для обучения CPU вполне подходит. Для реального обучения моделей нужен GPU.

**Тип:** Практика  
**Языки:** Python  
**Предварительные требования:** Фаза 0, Урок 01  
**Время:** ~45 минут

## Цели обучения

- Проверить доступность локального GPU с помощью `nvidia-smi` и CUDA API в PyTorch
- Настроить Google Colab с GPU T4 для бесплатных облачных экспериментов
- Сравнить производительность умножения матриц на CPU и GPU и измерить ускорение
- Оценить максимальный размер модели, помещающейся в VRAM, используя правило fp16

## Проблема

Большинство уроков фаз 1–3 нормально работают на CPU. Но как только вы начинаете обучать CNN, трансформеры или LLM (фазы 4+), вам потребуется GPU-ускорение. Обучение, которое занимает 8 часов на CPU, может занять 10 минут на GPU.

У вас есть три варианта: локальный GPU, облачный GPU или Google Colab (бесплатно).

## Концепция

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## Практика

### Вариант 1: локальный NVIDIA GPU

Проверьте, есть ли он у вас:

```bash
nvidia-smi
```

Установите PyTorch с CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Вариант 2: Google Colab

1. Перейдите на https://colab.research.google.com
2. Runtime → Change runtime type → T4 GPU
3. Выполните `!nvidia-smi` для проверки

Загружайте ноутбуки из этого курса напрямую в Colab.

### Вариант 3: облачный GPU

Для Lambda Labs, RunPod или Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Нет GPU? Не проблема.

Большинство уроков работают и на CPU. В уроках, где нужен GPU, это будет явно указано, а также будут даны ссылки на Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Практика: сравнение GPU и CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Упражнения

1. Запустите бенчмарк выше и сравните время работы CPU и GPU
2. Если у вас нет GPU, выполните это в Google Colab и сравните результаты
3. Проверьте объем памяти GPU и оцените максимальный размер модели, которую можно загрузить (правило: 2 байта на параметр для fp16)

## Ключевые термины

| Термин | Как обычно говорят | Что это означает на самом деле |
|------|----------------|----------------------|
| CUDA | «Программирование под GPU» | Платформа параллельных вычислений NVIDIA, позволяющая запускать код на GPU |
| VRAM | «Память GPU» | Видеопамять GPU, отдельная от системной RAM. Ограничивает размер модели |
| fp16 | «Половинная точность» | 16-битное число с плавающей точкой, использующее вдвое меньше памяти по сравнению с fp32 |
| Tensor Core | «Быстрое железо для матриц» | Специализированные ядра GPU для умножения матриц, работающие в 4–8 раз быстрее обычных |
