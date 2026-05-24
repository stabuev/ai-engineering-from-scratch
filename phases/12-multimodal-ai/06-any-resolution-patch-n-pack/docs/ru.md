# Any-Resolution Vision: Patch-n'-Pack и NaFlex

> Реальные изображения не являются квадратами 224x224. Чек имеет формат 9:16, график — 16:9, медицинский снимок может быть 4096x4096, мобильный скриншот — 9:19.5. Ответ VLM до 2024 года — изменить размер всего до фиксированного квадрата — выбрасывал сигнал, на котором работают OCR, понимание документов и высокоразрешающий разбор сцен. NaViT (Google, 2023) показал, что можно упаковывать патчи переменного разрешения в один transformer batch с block-diagonal masking. Qwen2-VL's M-RoPE (2024) полностью отказался от абсолютных позиционных таблиц. LLaVA-NeXT's AnyRes разбивал изображения высокого разрешения на базовое изображение + подизображения. SigLIP 2's NaFlex variant (2025) теперь является encoder по умолчанию для открытых VLM, которым нужен один checkpoint для всех соотношений сторон. Этот урок реализует patch-n'-pack от начала до конца.

**Тип:** Практика
**Языки:** Python (stdlib, patch packer + block-diagonal mask)
**Предварительные требования:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Время:** ~120 минут

## Цели обучения

- Упаковывать патчи из batch изображений переменного разрешения в одну последовательность и строить block-diagonal attention mask.
- Выбирать между AnyRes tiling (LLaVA-NeXT), NaFlex (SigLIP 2) и M-RoPE (Qwen2-VL) для заданной задачи.
- Вычислять бюджеты токенов для OCR, графиков и фотографий без изменения размера.
- Называть три режима отказа square-resize: сжатый текст, обрезанный контент, потраченные впустую токены на padding.

## Проблема

Transformers ожидают последовательность. Batch — это стек последовательностей одинаковой длины. Если ваши изображения имеют размер 224x224, вы каждый раз получаете 196 patch tokens, padding не нужен, задача решена. Обучайтесь на 224, выполняйте inference на 224 и больше никогда не думайте о разрешении.

Мир так не работает. Документы имеют портретную ориентацию (8.5x11 inches, примерно 2:3). Скриншоты графиков — ландшафтные (16:9). Чеки высокие и узкие (1:3). Медицинские изображения поставляются в 2048x2048 или больше. Скриншоты мобильных устройств — 1170x2532 (0.46:1).

Три варианта до 2024 года и почему каждый из них ломается:

1. Изменить размер до фиксированного квадрата (224x224 or 336x336). Сжатие искажает текст и лица. Уменьшение размера уничтожает подписи графиков и OCR-контент. Это было стандартной практикой до LLaVA-1.5.
2. Обрезать до фиксированного соотношения сторон. Вы выбрасываете большую часть изображения, а выбор места обрезки сам по себе является задачей computer vision.
3. Дополнить padding до самой длинной стороны. Это исправляет искажение, но для портретных изображений тратит 50%+ токенов на padding. Quadratic attention cost применяется ко всем этим pad tokens.

Ответ 2024-2025: позволить transformer обрабатывать патчи в родном разрешении изображения и понять, как упаковать неоднородный batch в одну последовательность без потраченных впустую вычислений.

## Концепция

### NaViT and patch-n'-pack

NaViT (Dehghani et al., 2023) был статьей, которая показала, что это работает в масштабе. Идея механическая:

1. Для каждого изображения в batch вычислить его родную patch grid при выбранном patch size (скажем, 14).
2. Развернуть патчи каждого изображения в собственную последовательность переменной длины.
3. Конкатенировать патчи всех изображений в одну длинную последовательность для batch.
4. Построить block-diagonal attention mask, чтобы патчи image A attended только внутри image A.
5. Передавать позиционную информацию для каждого патча (2D RoPE или fractional position embeddings).

Batch из трех изображений 336x336 (576 tokens), 224x224 (256 tokens) и 448x336 (768 tokens) становится одной последовательностью из 1600 токенов с block-diagonal mask 1600x1600. Без padding. Без потраченных впустую вычислений. Transformer обрабатывает произвольные соотношения сторон.

NaViT также ввел fractional patch dropping во время обучения — случайно отбрасывать 50% патчей по всему batch, — что одновременно регуляризует и ускоряет обучение. SigLIP 2 унаследовал это.

### AnyRes (LLaVA-NeXT)

LLaVA-NeXT's AnyRes — прагматичная альтернатива. Для изображения высокого разрешения и фиксированного encoder (CLIP или SigLIP на 336) изображение разбивается на tiles:

1. Выбрать grid layout из предопределенного набора — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), etc. — который лучше всего соответствует соотношению сторон изображения.
2. Разбить полное изображение на эту сетку; каждый tile становится crop 336x336.
3. Также создать thumbnail: все изображение, измененное до 336x336, как token глобального контекста.
4. Пропустить каждый tile через frozen 336-encoder. Конкатенировать tile tokens + thumbnail tokens.

Для изображения 672x672 с grid 2x2 плюс thumbnail: 4 * 576 + 576 = 2880 visual tokens. Дорого, но эффективно — LLM видит и локальные детали, и глобальный контекст.

AnyRes — предпочтительный путь, когда ваш encoder frozen и поддерживает только одно разрешение. Он резко увеличивает число токенов для больших изображений (изображение 1344x1344 с grid 4x4 — это 9216 + 576 ≈ 9800 tokens, что заполняет большую часть 8k LLM context).

### M-RoPE (Qwen2-VL)

Qwen2-VL ввел Multimodal Rotary Position Embedding. Вместо fractional positions из NaViT или tile-and-thumbnail из AnyRes каждый патч несет 3D-позицию (temporal, height, width). Повороты query/key обрабатывают произвольные H, W и temporal length.

M-RoPE поставляет native dynamic resolution без переобучения. На inference вы подаете любое HxW изображение, patch embedder производит H/14 x W/14 tokens, каждый token получает свою позицию (t=0, r=row, c=col), RoPE вращает attention с правильными частотами, готово. Qwen2.5-VL и Qwen3-VL продолжают это. InternVL3's V2PE — та же идея с variable encoding per modality.

В отличие от AnyRes, M-RoPE использует O(H x W / P^2) tokens в native resolution — без мультипликативных накладных расходов от tiles. В отличие от NaViT, он все еще ожидает одно изображение на forward. Batching across resolutions все равно требует patch-n'-pack сверху.

### NaFlex (SigLIP 2)

NaFlex — это native-flex mode checkpoint SigLIP 2. Одна модель обслуживает несколько sequence lengths (256, 729, 1024 tokens) на inference. Внутри она использует NaViT-style patch-n'-pack во время обучения и absolute fractional positions per patch. Главное преимущество: один checkpoint, а token budget на inference выбирается по задаче.

Для semantic task (classification, retrieval) — 256 tokens. Для OCR или понимания графиков — 1024 tokens. Без переобучения.

### The packing mask

Block-diagonal mask — место, где спотыкается большинство реализаций. Для packed sequence длины `N_total`, покрывающей изображения `i=0..B-1` с длинами `n_i`, mask `M` формы `(N_total, N_total)` равна 1, если оба индекса находятся в блоке одного изображения, иначе 0. Ее можно построить из списка cumulative length:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

В PyTorch это одна строка с `torch.block_diag` или explicit gather. Путь variable-length в FlashAttention (`cu_seqlens`) полностью пропускает mask и выполняет attention внутри sequences, используя напрямую cumulative-length tensor — примерно в 10 раз быстрее dense mask для типичных batch.

### Token budgets

Выбирайте стратегию по задаче:

- OCR / документы: 1024-4096 tokens. SigLIP 2 NaFlex на 1024 или AnyRes 3x3 + thumbnail.
- Графики и UI: 729-1024 tokens на 384-448 native. Qwen2.5-VL dynamic resolution с max pixels cap.
- Естественные фотографии: 256-576 tokens достаточно. Downstream LLM видит достаточно. Платите за токены там, где плотность контента высока.
- Видео: 64-128 tokens per frame после spatial pooling, 2-8 FPS. Lesson 12.17 covers this.

Производственное правило 2026 года: выбрать per-task max-pixels cap, кодировать в native aspect ratio до этого cap, упаковывать batch и пропускать padding. Qwen2.5-VL предоставляет `min_pixels` and `max_pixels` именно для этой ручки.

## Использование

`code/main.py` реализует patch-n'-pack для heterogeneous batch изображений с integer pixel coordinates. Он:

- Принимает список размеров изображений (H, W).
- Вычисляет длину patch sequence каждого изображения при patch size 14.
- Упаковывает их в одну последовательность общей длины `sum(n_i)`.
- Строит block-diagonal attention mask (dense, for clarity).
- Сравнивает packed cost с square-resize и AnyRes tiling.
- Печатает token budget table для mixed batch (receipt, chart, screenshot, photo).

Запустите его. Получающиеся числа объясняют, почему каждый открытый VLM 2026 года использует patch-n'-pack.

## Результат

Этот урок создает `outputs/skill-resolution-budget-planner.md`. Для workload со смешанными соотношениями сторон (OCR, charts, photos, video frames) и total-token budget он выбирает правильную стратегию (NaFlex, AnyRes, M-RoPE или fixed-square) и выдает per-request configuration. Используйте этот skill, когда sizing a VLM for a product — он предотвращает тихий 10x token blowup, который убивает latency budgets.

## Упражнения

1. Чек имеет размер 600x1500 (1:2.5). При patch size 14 сколько native-resolution tokens? Сколько после square-resize to 336? Что на практике сильнее теряет OCR accuracy?

2. Постройте block-diagonal mask для batch из четырех изображений с длинами 256, 576, 729, 1024. Проверьте, что attention matrix имеет размер 2585x2585 и ровно `256^2 + 576^2 + 729^2 + 1024^2` ненулевых элементов.

3. Для изображения 1792x896 при patch 14 сравните: (a) square-resize to 336 then encode, (b) AnyRes 2x1 + thumbnail, (c) M-RoPE at native. Что использует меньше всего токенов? Что сохраняет больше всего деталей?

4. Реализуйте fractional patch dropping: given a packed sequence, drop 50% of tokens uniformly at random, and update the block-diagonal mask accordingly. Измерьте изменение sparsity mask.

5. Прочитайте Section 3.2 of the Qwen2-VL paper (arXiv:2409.12191). В двух предложениях опишите, что контролируют `min_pixels` and `max_pixels` и почему важны обе границы.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT-style packing" | Конкатенация patch sequences переменной длины из разных изображений в одно batch dimension |
| Block-diagonal mask | "Packing mask" | Attention mask, которая ограничивает патчи каждого изображения attention только к самим себе, а не к соседям в pack |
| AnyRes | "LLaVA-NeXT tiling" | Разбить high-res image на grid фиксированных tiles плюс global thumbnail; кодировать каждый tile фиксированным encoder |
| NaFlex | "SigLIP 2 native-flex" | Единый checkpoint SigLIP 2, обслуживающий бюджеты 256/729/1024-token на inference без переобучения |
| M-RoPE | "Multimodal RoPE" | 3D rotary position encoding (time, row, column), обрабатывающий произвольные H, W, T без position tables |
| cu_seqlens | "FlashAttention packing" | Cumulative-length tensor, который FlashAttention varlen path использует вместо dense block-diagonal mask |
| min_pixels / max_pixels | "Resolution bounds" | Per-request knobs Qwen2.5-VL, ограничивающие token count для очень маленьких или очень больших inputs |
| Visual token budget | "How many tokens per image" | Приблизительное число patch tokens, emitted per image; задает prompt budget и attention cost для LLM |

## Дополнительное чтение

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
