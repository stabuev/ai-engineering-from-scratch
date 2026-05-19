# Python-окружения

> Dependency hell существует. Виртуальные окружения — лекарство от него.

**Тип:** Практика  
**Языки:** Python  
**Предварительные требования:** Фаза 0, Урок 01  
**Время:** ~30 минут

## Цели обучения

- Создавать изолированные окружения через `uv`, `venv` или `conda`
- Писать `pyproject.toml` и lockfiles для воспроизводимости
- Диагностировать типичные проблемы: глобальные установки, смешивание pip/conda, несовместимость CUDA
- Использовать стратегию отдельных окружений для разных фаз курса

## Проблема

Вы устанавливаете PyTorch 2.4 для одного проекта, а другому нужен PyTorch 2.1. Обновляете глобально — ломается первый проект. Откатываете назад — ломается второй.

Это и есть dependency hell.

## Концепция

Каждый проект должен иметь собственное изолированное окружение и собственные зависимости.

## Практика

### Вариант 1: uv (рекомендуется)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

cd your-project
uv venv
source .venv/bin/activate
```

Установка пакетов:

```bash
uv pip install torch numpy
```

### Вариант 2: venv

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch numpy
```

### Вариант 3: conda

Используйте conda, если вам нужны специфические версии CUDA или системных библиотек.

```bash
conda create -n myproject python=3.12
conda activate myproject

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## pyproject.toml

```toml
[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
]
```

## Lockfiles

Lockfile фиксирует точные версии зависимостей и гарантирует воспроизводимость окружения.

```bash
uv add numpy
```

## Типичные ошибки

### 1. Глобальная установка

```bash
pip install torch  # Плохо
```

```bash
source .venv/bin/activate
pip install torch  # Хорошо
```

### 2. Смешивание pip и conda

Старайтесь не использовать `pip install` внутри conda-окружения.

### 3. Забытая активация окружения

```bash
python train.py
```

убедитесь, что prompt выглядит так:

```text
(.venv) $
```

### 4. Коммит `.venv` в git

```bash
echo ".venv/" >> .gitignore
```

### 5. Несовместимость CUDA

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

Версия CUDA в PyTorch должна быть совместима с драйвером GPU.

## Упражнения

1. Создайте виртуальное окружение и проверьте его работу
2. Установите разные версии NumPy в двух окружениях
3. Напишите `pyproject.toml` для проекта с PyTorch и Anthropic SDK
4. Попробуйте глобальную установку пакета и посмотрите, куда он установится

## Ключевые термины

| Термин | Что это означает |
|------|----------------|
| Virtual environment | Изолированное Python-окружение |
| Lockfile | Фиксированные версии зависимостей |
| pyproject.toml | Основной конфигурационный файл Python-проекта |
| CUDA mismatch | Несовместимость версий CUDA |
