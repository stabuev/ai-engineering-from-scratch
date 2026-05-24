# LLaVA-OneVision: Single-Image, Multi-Image, Video в одной модели

> До LLaVA-OneVision (Li et al., August 2024) мир open-VLM имел отдельные линии: LLaVA-1.5 для single images, multi-image models вроде Mantis и VILA, video models вроде Video-LLaVA и Video-LLaMA. Каждая побеждала на своем benchmark и проваливалась на остальных. LLaVA-OneVision утверждала, что один curriculum может обучить одну модель доминировать во всех трех сценариях, а emergent task-transfer effects (single-image skills exported to video, multi-image reasoning exported to single-image) превосходят сумму specialists. Recipe обманчиво прост: visual-token budget, который остается постоянным между сценариями, плюс явный curriculum, двигающийся от single-image к OneVision (multi-image) и затем к video. Этот урок разбирает budget, curriculum и emergent behaviors.

**Тип:** Практика
**Языки:** Python (stdlib, token budget solver + curriculum planner)
**Предварительные требования:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Время:** ~180 минут

## Цели обучения

- Проектировать visual-token budget, который остается постоянным для single-image, multi-image и video inputs.
- Упорядочивать training curriculum, который переносит skills из single-image в video без catastrophic forgetting.
- Объяснять, почему одна модель превосходит specialists при том же parameter count, если curriculum сделан правильно.
- Называть три emergent capabilities, о которых сообщает LLaVA-OneVision: multi-camera reasoning, set-of-mark prompting, iPhone-screenshot agent.

## Проблема

Image, multi-image и video по-разному нагружают модель.

Single-image требует high-resolution tokens (AnyRes, ~2880 visual tokens), чтобы ловить OCR и мелкие детали. Budget per sample: one image, 2880 tokens.

Multi-image требует несколько изображений среднего разрешения (~576 tokens each), чтобы reasoning across images помещался в context. Budget per sample: 4-8 images, 576 each, 2300-4600 tokens.

Video требует много frames в low resolution (~196 tokens per frame after pooling), чтобы захватить temporal dynamics. Budget per sample: 8-32 frames, 196 each, 1600-6200 tokens.

Если вы обучаете отдельные модели, вы выбираете один budget. Если вы обучаете одну модель, budget должен масштабироваться осмысленно между сценариями, не взрывая context.

До OneVision стандартный ответ был "train one scenario, ignore the others." Video-LLaVA дооснастил image model видео через extra training stages. LLaVA-NeXT добавил multi-image support with tiling. Ничто не обрабатывало все три чисто.

## Концепция

### The OneVision token budget

LLaVA-OneVision выбирает unified visual-token budget примерно 3000-4000 tokens per sample, распределенный по-разному для каждого сценария:

- Single image: AnyRes-9 (3x3 tiles + thumbnail), каждый tile at 384 with 729 patches, aggressive bilinear pooling 2x2 → 182 per tile. Total: 9 * 182 + 182 = 1820 tokens. Или AnyRes-4 at 729-per-tile = 2916 + 729.
- Multi-image: каждое изображение at moderate resolution (384, no tiling), 729 tokens with no pooling. Budget 6 images → 4374 tokens.
- Video: 32 frames at 384 resolution with aggressive 3x3 bilinear pool → 81 tokens per frame. Total: 32 * 81 = 2592 tokens.

Распределение поддерживает примерно constant total tokens. LLM никогда не видит batch, который взрывает context. Encoder производит разную geometry по сценариям, но LLM потребляет тот же budget.

### The three-stage curriculum

LLaVA-OneVision обучается в три stage:

1. Single-image SFT (stage SI). Все data — single-image-plus-text. Обучение на high-resolution AnyRes input. Это учит perception, OCR и fine-grained understanding. Использует LLaVA-NeXT data plus OneVision-specific single-image data.
2. OneVision SFT (stage OV). Mix single-image + multi-image + video (uniformly sampled frames). Обучение на unified token budget. Это учит модель обрабатывать heterogeneous batch shapes. No weight reset — продолжает с stage SI.
3. Task transfer (stage TT). Продолжение с target task mix, обычно более тяжелым по multi-image или video в зависимости от product. Optional fine-tune for deployment.

Критично: порядок curriculum важен. Training video-first или multi-image-first дает хуже image performance, чем single-image-first, даже с теми же data. Статья явно ablates this.

### Why curriculum works

Single-image training строит perceptual base. Patch tokens несут fine-grained visual features; LLM учится интегрировать их с text. Multi-image и video вводят structural challenges (which image is which, what happened first), которые трудно выучить без сильной perceptual base.

Если обучать все сценарии с нуля вместе, модель недообучает perception (limited single-image data per batch) и переобучает structure (много multi-image / video data). Результат: модель следует cross-image reasoning patterns, но визуально поверхностна.

Curriculum ordering дает perception strength из stage SI, затем compositional/temporal reasoning из stage OV, не теряя ни то ни другое.

### Emergent cross-scenario skills

Статья LLaVA-OneVision сообщает о трех emergent capabilities:

1. Multi-camera reasoning. Обучена на multi-image + video separately; на inference ее просят рассуждать о multi-camera driving scene. Модель правильно интегрирует views, хотя никогда не видела такой exact format in training.
2. Set-of-mark prompting. Пользователь аннотирует объекты на изображении numbered marks; модель рассуждает о "what is mark 3 doing relative to mark 7." Не обучалась ни на marks, ни на annotation; learned from the combination of spatial grounding + multi-image reference.
3. iPhone-screenshot agent. Пользователь дает screenshot iPhone screen и просит спланировать next click. Обучалась на UI screenshots, video of user workflows и multi-image before/after pairs. Generalizes to the agent use case.

Это не trained tasks; они emerge from the curriculum's compositional structure.

### Visual-token pooling

Token budget требует pooling. OneVision использует bilinear interpolation на 2D patch grid: 24x24 = 576 patches становится 12x12 = 144 (2x factor) или 8x8 = 64 (3x factor). Pooling выполняется в patch-grid space, а не token space, чтобы сохранить locality.

Выбор pooling factor per scenario сам является hyperparameter. Less pooling = more tokens = richer representation. More pooling = fewer tokens = помещается больше frames / images.

### LLaVA-OneVision-1.5

Follow-up 2025 года (LLaVA-OneVision-1.5, arXiv 2509.23661) является "fully open" в training data, model weights и code. Закрывает proprietary gap на некоторых benchmarks и democratizes the recipe. Тот же curriculum, больше data, лучший base LLM. Без architecture change.

### Contrast with Qwen2.5-VL

Qwen2.5-VL (Lesson 12.09) делает другие выборы. Он использует M-RoPE и dynamic FPS вместо fixed pooling. Его budget масштабируется с input — 1-minute video использует больше tokens, чем 5-second video. LLaVA-OneVision фиксирует budget и масштабирует pooling. Оба подхода работают; они обменивают configurability на predictability.

## Использование

`code/main.py` — это curriculum and budget planner для OneVision-style VLM. Для заданного token budget per sample и target scenario mix (say 40% single-image, 30% multi-image, 30% video) он:

- Распределяет resolution, pooling factor и frames per scenario.
- Проверяет, что каждый scenario помещается в shared budget.
- Сообщает expected token count, LLM FLOPs и какие scenarios are under-tokenized.
- Печатает stage-by-stage training schedule.

Используйте его, чтобы планировать OneVision fine-tune или sanity-check per-request cost VLM deployment.

## Результат

Этот урок создает `outputs/skill-onevision-budget-planner.md`. Для target task distribution и per-sample budget он выдает AnyRes factor, per-frame pooling, video frame count и curriculum stage weights. Используйте это всякий раз, когда train or fine-tune a unified-scenario VLM.

## Упражнения

1. Ваш product поддерживает 80% single-image, 10% multi-image (2-4 images), 10% video (8-16 frames). Спроектируйте token budget. Куда бы вы поместили extra budget, сэкономленный от отсутствия heavy multi-image?

2. Прочитайте LLaVA-OneVision Section 4.3 (emergent capabilities). Предложите fourth emergent skill, который curriculum, вероятно, открыл бы, но paper не сообщил.

3. Поменяйте curriculum order — train multi-image first, then single-image, then video. Предскажите, какие benchmarks degrade и почему.

4. Paper reports video benchmarks trained on only 8 frames per sample. Обобщается ли это на 30-second videos at inference? Что ломается первым — token budget или temporal reasoning?

5. Bilinear pooling of 24x24 patches to 12x12 is a 4x reduction per dim. Реализуйте pooling in stdlib Python и проверьте, что mean over each 2x2 block matches the bilinear output.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| OneVision scenario | "Single-image, multi-image, or video" | Одна из трех input shapes, которые unified VLM обрабатывает; budget остается постоянным across |
| Token budget | "How many tokens per sample" | Total visual tokens, которые LLM видит per training / inference sample, обычно 3000-4000 |
| Curriculum | "Training order" | Stage ordering (single-image → multi-image → video), выбранный для emergent transfer |
| Bilinear pooling | "Token shrink" | Применение bilinear interpolation к patch grid (2D), чтобы уменьшить token count при сохранении locality |
| Emergent skill | "Not trained, still works" | Capability, появляющаяся на inference без matching training data благодаря curriculum composition |
| AnyRes-k | "k-tile setup" | k sub-tiles фиксированного разрешения плюс one thumbnail, typical k ∈ {4, 9} |
| Task transfer | "Cross-scenario generalization" | Skills, learned on single-image, которые применяются к video (and vice versa) через shared backbone |

## Дополнительное чтение

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
