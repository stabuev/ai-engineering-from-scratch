# Vision Transformers (ViT)

> Изображение — это сетка patches. Предложение — это сетка tokens. Один и тот же transformer ест и то, и другое.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 7 · 05 (полный transformer), Фаза 4 · 03 (CNN), Фаза 4 · 14 (введение в Vision Transformers)
**Время:** ~45 минут

## Цели обучения

- Разбивать изображение на патчи, линейно их эмбеддить, добавлять токен [CLS] и подавать в стандартный энкодер трансформера.
- Объяснять, от какого индуктивного смещения отказывается ViT по сравнению с CNN и почему ему понадобился масштаб.
- Размещать DINOv2, SigLIP и Swin как значимые варианты.

## Проблема

До 2020 года computer vision означал convolutions. Каждый SOTA на ImageNet, COCO и detection benchmarks использовал CNN backbone. Transformers были для языка.

Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words" — показали, что convolutions можно полностью убрать. Разрежьте image на fixed-size patches, linearly project каждый patch в embedding, подайте sequence в vanilla transformer encoder. При достаточном scale (ImageNet-21k pretraining или больше) ViT догоняет или обходит ResNet-based models.

ViT стал началом паттерна 2026 года: одна architecture, много modalities. Whisper tokenizes audio. ViT tokenizes images. Action tokens для robotics. Pixel tokens для video. Transformer не заботится о типе данных — дайте sequence, и он выучит.

К 2026 году ViT и потомки (DeiT, Swin, DINOv2, ViT-22B, SAM 3) занимают большую часть vision. CNNs все еще выигрывают на edge devices и latency-sensitive tasks. Во всем остальном где-то в stack есть ViT.

## Концепция

![Image → patches → tokens → transformer](../assets/vit.svg)

### Шаг 1 — patchify

Разделите image `H × W × C` на sequence `N × (P·P·C)` flat patches. Типичный setup: image `224 × 224`, patches `16 × 16` → 196 patches по 768 values.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch size — главный рычаг. Smaller patches = больше tokens, лучше resolution, quadratic attention cost. Larger patches = грубее, дешевле.

### Шаг 2 — linear embedding

Одна learned matrix проецирует каждый flat patch в `d_model`. Это эквивалент convolution с kernel size `P` и stride `P`. В PyTorch это буквально `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`.

### Шаг 3 — добавьте `[CLS]` token в начало и positional embeddings

- Добавьте learnable `[CLS]` token в начало. Его final hidden state — image representation для classification.
- Добавьте learnable positional embeddings (оригинальный ViT) или sinusoidal 2D (поздние варианты).
- В 2024+ RoPE расширили до 2D position, иногда без explicit embeddings.

### Шаг 4 — стандартный transformer encoder

Стек L blocks `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Идентично BERT. Без vision-specific layers. Это педагогический punchline статьи.

### Шаг 5 — head

Для classification: `[CLS]` hidden state → linear → softmax. Для DINOv2 или SAM `[CLS]` часто отбрасывают и используют patch embeddings напрямую.

### Варианты, которые имели значение

| Модель | Год | Изменение |
|--------|-----|-----------|
| ViT | 2020 | Оригинал. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; обучается только на ImageNet-1k. |
| Swin | 2021 | Hierarchical с shifted windows. Fixed sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (без labels). Лучшие general vision features. |
| ViT-22B | 2023 | 22B params; scaling laws применимы. |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### Почему на это ушло время

ViT требует *очень много* данных, чтобы сравняться с CNNs, потому что у него нет CNN inductive biases (translation invariance, locality). Без >100M labeled images или сильного self-supervised pretraining CNNs выигрывают при matched compute. DeiT исправил это в 2021 distillation tricks; DINOv2 окончательно исправил в 2023 через self-supervision.

## Соберите это

См. `code/main.py`. Pure-stdlib patchify + linear embedding + sanity checks. Без training — ViT реалистичного размера требует PyTorch и часы GPU time.

### Шаг 1: fake image

Изображение RGB 24 × 24 как список rows из `(R, G, B)` tuples. Используем patches 6×6 → 16 patches, embedding vector длины 108 для каждого.

### Шаг 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster order: row-major по grid. Каждый ViT использует этот ordering.

### Шаг 3: linear embedding

Умножьте каждый flat patch на random matrix `(patch_flat_size, d_model)`. Проверьте, что output shape равна `(N_patches + 1, d_model)` после prepending `[CLS]`.

### Шаг 4: посчитайте параметры для реалистичного ViT

Напечатайте param count для ViT-Base: 12 layers, 12 heads, d=768, patch=16. Сравните с ResNet-50 (~25M). ViT-Base дает ~86M. ViT-Large ~307M. ViT-Huge ~632M.

## Используйте это

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings — default для image features в 2026 году.** Freeze backbone, train tiny head. Работает для classification, retrieval, detection, captioning. Checkpoints Meta DINOv2 превосходят CLIP почти на всех non-text vision tasks.

**Выбор patch size.** Small models используют 16×16 (ViT-B/16). Dense prediction (segmentation) использует 8×8 или 14×14 (SAM, DINOv2). Very large models используют 14×14.

## Доведите до поставки

См. `outputs/skill-vit-configurator.md`. Skill выбирает ViT variant и patch size для новой vision task с учетом dataset size, resolution и compute budget.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Проверьте, что number of patches равно `(H/P) * (W/P)`, а flat patch dimension равно `P*P*C`.
2. **Средне.** Реализуйте 2D sinusoidal positional embeddings — два независимых sinusoidal codes для `row` и `col` каждого patch, concatenated. Подайте их в tiny PyTorch ViT и сравните accuracy с learnable positional embeddings на CIFAR-10.
3. **Сложно.** Постройте 3-layer ViT (PyTorch), обучите на 1,000 MNIST images с patches 4×4. Измерьте test accuracy. Затем добавьте DINOv2 pretraining на тех же 1,000 images (simplified: train encoder to predict patch embeddings from masked patches). Улучшится ли accuracy?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|------------|-------------------------------|
| Patch | "Токен vision transformer" | Flat vector pixel values для области image `P × P × C`. |
| Patchify | "Нарезать и выпрямить" | Нарезать image на non-overlapping patches и flatten each to vector. |
| `[CLS]` token | "Сводка изображения" | Prepended learnable token; его final embedding — image representation. |
| Inductive bias | "Что предполагает модель" | У ViT меньше priors, чем у CNNs; нужны дополнительные data. |
| DINOv2 | "Self-supervised ViT" | Обучен без labels через image augmentation + momentum teacher. Лучшие general image features в 2026. |
| SigLIP | "Наследник CLIP" | ViT + text encoder with sigmoid contrastive loss; лучше CLIP при matched compute. |
| Swin | "Windowed ViT" | Hierarchical ViT with local attention + shifted windows; sub-quadratic. |
| Register tokens | "Трюк 2023 года" | Несколько extra learnable tokens, поглощающих attention sinks; улучшают DINOv2 features. |

## Дополнительное чтение

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — статья ViT.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — register-token fix для DINOv2.
