# Vision Transformers и примитив patch-token

> До любой мультимодальности изображение должно стать последовательностью токенов, которую может обработать трансформер. Статья ViT 2020 года ответила на это патчами 16x16 пикселей, линейной проекцией и позиционным эмбеддингом. Пять лет спустя каждая frontier-модель 2026 года (Claude Opus 4.7 at 2576px native, Gemini 3.1 Pro, Qwen3.5-Omni) все еще начинается так же — энкодер изменился с ViT на DINOv2 и SigLIP 2, добавились register tokens, позиционная схема стала 2D-RoPE, но сам примитив сохранился. В этом уроке мы разбираем patch-token pipeline от начала до конца и строим его на stdlib Python, чтобы у остальной Phase 12 была конкретная ментальная модель для "visual tokens."

**Тип:** Изучение
**Языки:** Python (stdlib, patch tokenizer + geometry calculator)
**Предварительные требования:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Время:** ~120 минут

## Цели обучения

- Преобразовать изображение HxWx3 в последовательность patch tokens с корректным positional encoding.
- Вычислять длину последовательности, число параметров и FLOPs для ViT с заданными (patch size, resolution, hidden dim, depth).
- Назвать три улучшения, которые довели ViT от исследования 2020 года до production 2026 года: self-supervised pretraining (DINO / MAE), register tokens и native-resolution packing.
- Выбирать между CLS pooling, mean pooling и register tokens для downstream task.

## Проблема

Трансформеры работают с последовательностями векторов. Текст уже является последовательностью (байты или токены). Изображение — это 2D-сетка пикселей с тремя цветовыми каналами, а не последовательность. Если развернуть каждый пиксель, RGB-изображение 224x224 превратится в 150,528 токенов, а self-attention на такой длине непрактичен (квадратичная сложность по длине последовательности).

Подходы до 2020 года добавляли CNN feature extractor спереди: ResNet производит feature map 7x7 из 2048-мерных векторов, эти 49 токенов подаются в трансформер. Это работает, но наследует bias CNN (translation equivariance, локальные receptive fields) и теряет трансформерную способность масштабироваться.

Dosovitskiy et al. (2020) задали прямой вопрос: что если пропустить CNN? Разбить изображение на патчи фиксированного размера (например, 16x16 пикселей), линейно спроецировать каждый патч в вектор, добавить positional embedding и подать последовательность в vanilla transformer. На тот момент это было ересью — computer vision без сверток. При достаточном количестве данных (JFT-300M, затем LAION) подход обогнал ResNet на ImageNet и продолжил улучшаться.

К 2026 году примитив ViT стал бесспорной основой. Vision tower каждой open-weights VLM является каким-то потомком (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Вопрос уже не "использовать ли патчи?", а "какой patch size, какой resolution schedule, какая pretraining objective, какое positional encoding."

## Концепция

```mermaid
graph LR
  IMG["image"] --> P["patchify 16×16"]
  P --> E["linear patch embedding"]
  E --> CLS["prepend [CLS] + position emb"]
  CLS --> ENC["transformer encoder"]
  ENC --> O["patch tokens + pooled output"]
```

### Patches as tokens

Дано изображение `x` формы `(H, W, 3)` и размер патча `P`. Вы нарезаете изображение на сетку непересекающихся патчей `(H/P) x (W/P)`. Каждый патч — это куб пикселей `P x P x 3`. Разверните каждый куб в вектор `3 P^2`. Примените общую линейную проекцию `W_E` формы `(3 P^2, D)`, чтобы отобразить каждый патч в скрытую размерность модели `D`.

Для канонической конфигурации ViT-B/16:
- Resolution 224, patch size 16 → grid 14x14 → 196 patch tokens.
- Каждый патч содержит `16 x 16 x 3 = 768` значений пикселей, проецируемых в `D = 768`.
- Добавляется обучаемый токен `[CLS]` → длина последовательности 197.

Patch projection математически идентична 2D-свертке с kernel size `P`, stride `P` и `D` выходными каналами. Именно так production-код обычно это реализует — `nn.Conv2d(3, D, kernel_size=P, stride=P)`. Формулировка "linear projection" концептуальна; формулировка через kernel эффективна.

### Positional embeddings

У патчей нет внутреннего порядка — трансформер видит их как мешок. Ранние ViT добавляли обучаемый 1D positional embedding (один 768-мерный вектор на позицию, всего 197). Это работает, но привязывает модель к training resolution: на inference приходится интерполировать таблицу позиций, если меняется сетка.

Современные vision backbones используют 2D-RoPE (Qwen2-VL's M-RoPE, SigLIP 2's default) или факторизованные 2D-позиции. 2D-RoPE вращает query и key vectors на основе индекса патча (row, column), так что модель выводит относительную 2D-позицию из угла вращения. Таблица позиций не нужна. Модель обрабатывает произвольные размеры сетки на inference.

### CLS token, pooled output, and register tokens

Что является image-level representation? Сосуществуют три варианта:

1. `[CLS]` token. Обучаемый вектор добавляется в начало patch sequence. После всех transformer blocks скрытое состояние CLS token становится представлением изображения. Унаследовано от BERT. Используется в original ViT, CLIP.
2. Mean pool. Усреднение выходных hidden states patch tokens. Используется в SigLIP, DINOv2, большинстве современных VLMs.
3. Register tokens. Darcet et al. (2023) заметили, что ViT, обученные без явного sink token, развивают высоконормовые "artifact" patches, которые перехватывают self-attention. Добавление 4–16 обучаемых register tokens поглощает эту нагрузку и улучшает качество dense-prediction (segmentation, depth). DINOv2 и SigLIP 2 поставляются с registers.

Выбор важен для downstream tasks. CLS подходит для classification. Для VLMs, которые подают patch tokens в LLM, pooling полностью пропускается — каждый патч становится входным LLM-токеном. Registers отбрасываются перед handoff (это строительные леса, а не content).

### Pretraining: supervised, contrastive, masked, self-distilled

ViT 2020 года был предварительно обучен supervised classification на JFT-300M. Его быстро вытеснили:

- CLIP (2021): contrastive image-text на 400M пар. Lesson 12.02.
- MAE (2021, He et al.): маскировать 75% патчей, восстанавливать пиксели. Self-supervised, работает на чистых изображениях.
- DINO (2021) / DINOv2 (2023): self-distillation со student-teacher, без labels, без captions. DINOv2 ViT-g/14 2023 года — сильнейший purely-visual backbone и default для use cases с "dense features".
- SigLIP / SigLIP 2 (2023, 2025): CLIP с sigmoid loss и NaFlex для native aspect ratio. Доминирующий vision tower в open VLMs 2026 года (Qwen, Idefics2, LLaVA-OneVision).

Выбор pretraining определяет, в чем backbone силен: CLIP/SigLIP для semantic matching с текстом, DINOv2 для dense visual features, MAE как starting point для downstream finetuning.

### Scaling laws

ViT scaling (Zhai et al. 2022) установил, что качество ViT подчиняется предсказуемым законам по model size, data size и compute. При фиксированном compute:
- Bigger model + more data → better quality.
- Patch size — рычаг между sequence length и fidelity. Patch 14 (типичный для DINOv2/SigLIP SO400m) дает больше токенов на изображение, чем patch 16; лучше для OCR и dense tasks, хуже для скорости.
- Resolution — второй большой рычаг. Переход с 224 на 384 и 512 почти всегда помогает, но с квадратичной ценой в FLOPs.

ViT-g/14 (1B params, patch 14, resolution 224 → 256 tokens) и SigLIP SO400m/14 (400M params, patch 14) — два рабочих энкодера для open VLMs 2026 года.

### Parameter count for a ViT

Полный расчет находится в `code/main.py`. Для ViT-B/16 at 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Прикидывайте любой ViT так до загрузки checkpoint. Размер backbone задает нижнюю границу VRAM в любой downstream VLM.

### 2026 production config

Энкодер, с которым в 2026 году поставляется большинство open VLMs, — SigLIP 2 SO400m/14 at native resolution (NaFlex). У него:
- 400M parameters.
- Patch size 14, default resolution 384 → 729 patch tokens per image.
- Mean pool для image-level tasks; все 729 патчей идут в LLM для VQA.
- 4 register tokens, отбрасываемые перед LLM handoff.
- 2D-RoPE with image-level scaling для native aspect ratio.

Каждое решение в этой конфигурации восходит к статье, которую можно прочитать.

## Использование

`code/main.py` — это patch tokenizer и geometry calculator. Он принимает (image H, W, patch P, hidden D, depth L) и сообщает:

- Grid shape и sequence length после patching.
- Token sequence для синтетического toy image 8x8 пикселей (прохождение flatten + project path).
- Parameter count с разбивкой на patch embed, position embed, transformer blocks и head.
- FLOPs на forward pass при target resolution.
- Comparison table для ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Запустите его. Сопоставьте parameter counts с опубликованными числами. Поиграйте с patch size и resolution, чтобы почувствовать цену token-count.

## Результат

Этот урок создает `outputs/skill-patch-geometry-reader.md`. По конфигурации ViT (patch size, resolution, hidden dim, depth) он выдает token-count, parameter-count и оценку VRAM с обоснованием. Используйте этот skill каждый раз, когда выбираете vision backbone для VLM — он предотвращает сюрпризы вида "tokens exploded and my LLM context filled up".

## Упражнения

1. Вычислите длину patch-token sequence для Qwen2.5-VL при native input 1280x720 с patch size 14. Как это сравнивается с CLS-only representation?

2. Сколько токенов дает 1080p frame (1920x1080) при patch 14? При 30 FPS на 5-минутном видео сколько всего visual tokens? Что больше всего снижает cost: pooling, frame sampling или token merging?

3. Реализуйте mean pooling over patch tokens на pure Python. Проверьте, что mean-pool по 196 tokens DINOv2 output совпадает с тем, что возвращает `forward` модели, когда вы запрашиваете pooled embedding.

4. Прочитайте Section 3 статьи "Vision Transformers Need Registers" (arXiv:2309.16588). Опишите в двух предложениях, какой artifact поглощают registers и почему это важно для downstream dense prediction.

5. Измените `code/main.py`, чтобы поддержать patch-n'-pack: по списку изображений разных resolutions создать одну packed sequence и block-diagonal attention mask. Проверьте на Lesson 12.06, когда дойдете до него.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Patch | "16x16 pixel square" | Фиксированная непересекающаяся область входного изображения; становится одним токеном |
| Patch embedding | "Linear projection" | Общая обучаемая матрица (или Conv2d with stride=P), отображающая развернутые пиксели патча в D-мерные векторы |
| CLS token | "Class token" | Добавляемый в начало обучаемый вектор, final hidden state которого представляет все изображение; optional in 2026 |
| Register token | "Sink token" | Дополнительные обучаемые токены, поглощающие high-norm attention artifacts, которые ViT развивают во время pretraining |
| Position embedding | "Positional info" | Вектор или rotation на позицию, делающие sequence-order-aware; 2D-RoPE is the modern default |
| Grid | "Patch grid" | 2D-массив патчей `(H/P) x (W/P)` для заданных resolution и patch size |
| NaFlex | "Native flexible resolution" | Функция SigLIP 2: одна модель обслуживает несколько aspect ratios и resolutions без retraining |
| Backbone | "Vision tower" | Предобученный image encoder, whose patch-token outputs feed the LLM in a VLM |
| Pooling | "Image-level summary" | Стратегия преобразования patch tokens в один вектор: CLS, mean, attention pool или register-based |
| Patch 14 vs 16 | "Finer vs coarser grid" | Patch 14 дает больше токенов на изображение, лучшую fidelity для OCR, но медленнее; patch 16 — classic default |

## Дополнительное чтение

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — original ViT.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, self-supervised pretraining.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — self-distillation at scale, no labels.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register tokens and artifact analysis.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — the 2026 default vision tower.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — empirical scaling laws.
