# CLIP и контрастивное vision-language pretraining

> CLIP от OpenAI (2021) доказал одну идею, достаточно большую для следующих пяти лет: выровнять image encoder и text encoder в одном векторном пространстве, используя только шумные web image-caption pairs и contrastive loss. Ноль supervised labels. 400M пар. Получившееся embedding space делает zero-shot classification, image-text retrieval и подключается к каждой VLM 2026 года как vision tower. SigLIP 2 (2025) заменил softmax на sigmoid и масштабировался дальше CLIP с меньшей ценой. В этом уроке мы проходим математику от InfoNCE до sigmoid pairwise loss и строим training step на stdlib Python.

**Тип:** Практика
**Языки:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Предварительные требования:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Время:** ~180 минут

## Цели обучения

- Вывести InfoNCE loss из mutual information и реализовать численно стабильную vectorized version.
- Объяснить, почему sigmoid pairwise loss (SigLIP) масштабируется до batch 32768+ без all-gather overhead, которого требует softmax.
- Запустить zero-shot ImageNet classification, построив text templates (`a photo of a {class}`) и взяв argmax по cosine similarity.
- Назвать четыре рычага, которые дает CLIP / SigLIP pretraining: batch size, temperature, prompt template, data quality.

## Проблема

До CLIP vision был supervised. Соберите labeled datasets (ImageNet: 1.2M images, 1000 classes), обучите CNN, отправьте в production. Labels дороги, labels смещены к тому, о чем могут договориться разметчики, и labels не переносятся на новые tasks без finetuning.

В image-caption web есть миллиард с лишним loosely-labeled pairs бесплатно. Картинка golden retriever с alt text "my dog Max in the park" несет supervisory signal — текст описывает изображение. Вопрос: можно ли превратить это в полезное обучение?

Ответ CLIP: рассматривать image-caption pairs как matching task. Дана batch из N images и N captions; нужно научиться сопоставлять каждое изображение с его caption против N-1 distractors. Supervision: "эти две вещи принадлежат друг другу; эти N-1 — нет." Без class labels. Без human annotation. Только contrastive loss.

Получившееся embedding space умеет больше, чем то, на чем CLIP обучали. ImageNet zero-shot работает, потому что "a photo of a cat" оказывается рядом с картинками cats, которые никогда явно не размечались как cats. Это ставка, породившая каждую VLM 2026 года.

## Концепция

### The dual encoder

У CLIP две башни:

- Image encoder `f`: ViT или ResNet, выдает D-мерный вектор на изображение.
- Text encoder `g`: небольшой transformer, выдает D-мерный вектор на caption.

Обе башни нормализуют выходы до единичной длины. Similarity равна `cos(f(x), g(y)) = f(x)^T g(y)`, так как оба вектора unit-norm.

Для batch из N пар (image, caption) строится similarity matrix `S` формы `(N, N)`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

где `tau` — обучаемая temperature (CLIP initializes to 0.07; learned in log-space).

### InfoNCE loss

CLIP использует symmetric cross-entropy по строкам и столбцам:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

Это InfoNCE. Softmax в CE заставляет каждое изображение совпадать со своим caption сильнее, чем с любым другим caption в batch. "Negatives" — все остальные элементы batch. Bigger batches = more negatives = stronger signal. CLIP trained at batch 32k; scale matters.

### Temperature

`tau` управляет резкостью softmax. Low tau → sharp distribution, эффект hard negative mining. High tau → soft, вклад дают все samples. CLIP обучает log(1/tau), clipped to prevent collapse. SigLIP 2 фиксирует initial tau и использует learned bias вместо этого.

### Why sigmoid scales better (SigLIP)

Softmax требует синхронизации всей similarity matrix. В distributed training нужно all-gather каждого embedding на каждую replica, затем считать softmax. Это квадратично по world size для communication.

SigLIP заменяет softmax на element-wise sigmoid: для каждой пары `(i, j)` loss — binary classification "является ли это matching pair?" positive class labels находятся на диагонали, все остальное negative. Loss:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` если `i == j`, иначе 0. Loss каждой пары независим. All-gather не нужен. Каждый GPU считает свой local block и суммирует. SigLIP 2 дешево масштабируется до batch 32k-512k там, где CLIP потребовал бы пропорционально больше communication.

### Zero-shot classification

Дано N имен классов; для каждого класса строится text template:

```
"a photo of a {class}"
```

Embed each template with the text encoder. Embed your image with the image encoder. Argmax cosine similarity = predicted class. No training on the target classes.

Prompt templates matter. Оригинальная статья CLIP использовала 80 templates per class (plain, artistic, photo, painting, etc.) и усредняла embeddings. +3 ImageNet points. Modern usage typically picks one or two templates.

### Linear probes and finetuning

Zero-shot — baseline. Linear probe (обучить один linear layer поверх frozen CLIP features для target classes) обгоняет zero-shot на in-domain tasks. Full finetuning обгоняет linear probe на in-domain, но может вредить zero-shot transfer. Три режима с тремя trade-offs.

### SigLIP 2: NaFlex and dense features

SigLIP 2 (2025) добавляет:
- NaFlex: одна модель обрабатывает variable aspect ratios и resolutions.
- Лучшие dense features для segmentation и depth estimation, с прицелом на использование frozen backbone в VLMs.
- Multilingual: trained on 100+ languages, тогда как CLIP был English-only.
- 1B param scale, тогда как CLIP topped out at 400M.

В open VLMs 2026 года SigLIP 2 SO400m/14 — default vision tower. CLIP остается default для pure image-text retrieval, когда конкретное LAION-2B training distribution совпадает с вашим query pattern.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): та же идея, что CLIP, scale 1.8B pairs, 90% noisy. Доказал, что noisy data scales. OpenCLIP (LAION): open reproduction of CLIP on LAION-400M / 2B, multiple scales, основной open checkpoint. EVA-CLIP: initializes from masked image modeling; strong backbone for VLMs. BASIC: Google's CLIP+ALIGN hybrid. Все это одно семейство, различаются data и tuning.

### The zero-shot ceiling

CLIP-class models упираются примерно в 76% ImageNet zero-shot (CLIP-G, OpenCLIP-G). Дальше нужны либо much larger data (SigLIP 2 gets 80%+), либо architecture changes (supervised heads, more parameters). Benchmark насыщается; настоящая ценность — embedding space, которое потребляют downstream VLMs.

## Использование

`code/main.py` реализует:

1. Toy dual encoder (hash-based image features, text char features), чтобы увидеть форму InfoNCE без numpy.
2. InfoNCE loss на pure Python (numerical stability via log-sum-exp).
3. Sigmoid pairwise loss для сравнения.
4. Zero-shot classification routine: compute cosine similarity against a set of text prompts, argmax for prediction.

Запустите и посмотрите на loss curve. Absolute numbers toy; форма совпадает с тем, что выдает real CLIP trainer.

## Результат

Этот урок создает `outputs/skill-clip-zero-shot.md`. По набору images (via path) и списку target classes он строит text prompts с CLIP template, embeds both sides with a stated checkpoint (e.g., `openai/clip-vit-large-patch14`) и возвращает top-1 / top-5 predictions with similarity scores. Skill refuses to make claims about classes not in the prompt list.

## Упражнения

1. Реализуйте InfoNCE для batch из 4 пар вручную. Постройте 4x4 similarity matrix, выполните softmax, выберите diagonal, вычислите cross-entropy. Проверьте Python implementation against this hand calculation.

2. SigLIP использует bias parameter `b` дополнительно к temperature: `S'[i,j] = S[i,j]/tau + b`. Какую роль играет `b`, когда в batch большой class imbalance (намного больше negatives, чем positives per row)? Прочитайте SigLIP Section 3 (arXiv:2303.15343).

3. Постройте zero-shot classifier для cats vs dogs. Попробуйте два prompt templates: `a photo of a {class}` и `a picture of a {class}`. Измерьте accuracy на 100 test images. Ensemble templates beats single?

4. Вычислите communication cost softmax InfoNCE vs sigmoid pairwise для 512-GPU run at batch 32k. Что масштабируется как O(N), а что как O(N^2)? Cite SigLIP Section 4.

5. Прочитайте OpenCLIP scaling-laws paper (arXiv:2212.07143, Cherti et al.). Воспроизведите их conclusion for data scaling from the figures: при fixed model size, какова log-linear relationship между ImageNet zero-shot accuracy и training data size?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | Cross-entropy по batch similarity matrix; positive каждого item — paired item, negatives — все остальные |
| Sigmoid loss | "SigLIP loss" | Per-pair binary cross-entropy; no softmax, no all-gather, дешево масштабируется в distributed training |
| Temperature | "tau" | Scalar, масштабирующий logits перед softmax/sigmoid; controls sharpness of the distribution |
| Zero-shot | "no-finetune classification" | Использовать text prompts для построения class embeddings и классифицировать по cosine similarity; no training on target classes |
| Prompt template | "a photo of a ..." | Текстовый каркас вокруг class name; affects zero-shot accuracy by 1-5 points |
| Dual encoder | "Two-tower" | One image encoder + one text encoder, outputs in shared D-dim space |
| Hard negative | "Tough distractor" | Negative, достаточно похожий на positive, чтобы модель должна была работать для их разделения |
| Linear probe | "Frozen + one layer" | Обучить только linear classifier поверх frozen features; measures feature quality |
| NaFlex | "Native flexible resolution" | SigLIP 2 capability to ingest images at any aspect ratio and resolution without resizing |
| Temperature scaling | "log-parametrized tau" | CLIP parametrizes `log(1/tau)`, чтобы gradients behave; clips to prevent collapse to near-zero tau |

## Дополнительное чтение

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — the CLIP paper.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — multilingual + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — scale with noisy web data.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws.
