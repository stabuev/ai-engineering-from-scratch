# Видео-языковые модели: временные токены и grounding

> Видео — это не стопка фотографий. 5-секундный клип имеет причинный порядок, глаголы действий и время событий, которые модель изображений не может представить. Video-LLaMA (Zhang et al., июнь 2023) выпустила первую открытую video-LLM с аудиовизуальным grounding. VideoChat и Video-LLaVA масштабировали этот паттерн. К 2025 году TMRoPE в Qwen2.5-VL сократил разрыв с передовыми проприетарными моделями. Каждая система решала временные токены по-своему — Q-former на клип, concat-pool на кадр, TMRoPE на токен. В этом уроке разбираются паттерны, строится uniform-vs-dynamic frame sampler и проводится оценка на задачах temporal grounding.

**Тип:** Практика
**Языки:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Предварительные требования:** Phase 12 · 08 (LLaVA-OneVision)
**Время:** ~180 минут

## Цели обучения

- Объяснить, почему временное позиционное кодирование меняет качество video VLM независимо от vision encoder.
- Сравнить uniform, dynamic-FPS и event-driven frame sampling по tokens-per-second и точности grounding.
- Описать дизайны Q-former-per-clip (Video-LLaMA), pooled-per-frame (Video-LLaVA) и M-RoPE-per-token (Qwen2.5-VL).
- Назвать четыре видео-бенчмарка: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Проблема

1-минутное видео при 30 FPS — это 1800 кадров. При 196 визуальных токенах на кадр (ViT-B at 224) это 352k токенов — больше контекста любой LLM эпохи 2024 года.

Есть три стратегии сокращения:

1. Прореживать кадры (1-8 FPS в зависимости от контента).
2. Агрессивно пулить patch-токены каждого кадра (3x3 или 4x4 bilinear pool).
3. Сжимать через Q-former, который берет 16-кадровый клип и выдает 64 токена.

У каждого компромисс свой. Прореживание теряет временную детализацию. Пулинг теряет пространственную детализацию. Q-former теряет понемногу и то и другое, но экономит токены.

Временное позиционное кодирование — другая ось: как модель узнает, что frame 5 был до frame 6? Варианты включают простой 1D temporal RoPE (Video-LLaMA), обучаемые temporal embeddings (Video-LLaVA) и TMRoPE (Qwen2.5-VL, полноценное 3D).

## Концепция

### Video-LLaMA: Q-former per clip + audio branch

Video-LLaMA (2023) была первой открытой video-LLM. Архитектура:

- 16-кадровые клипы при 2 FPS (то есть 8 секунд).
- Per-frame ViT features -> Video Q-former, который cross-attends по всем 16 кадрам -> 32 learned queries -> LLM.
- Параллельная audio branch: waveform -> ImageBind audio encoder -> Audio Q-former -> 32 queries -> LLM.

Сильная сторона: совместное аудиовизуальное рассуждение. Слабость: фиксированная длина клипа, нет произвольного time grounding.

### VideoChat and Video-LLaVA

VideoChat сохранил идею Video-LLaMA, но убрал аудио и упростил систему. Video-LLaVA (Lin et al., 2023) обучила один visual encoder и на изображениях, и на видеокадрах ("alignment before projection"), получив единое представление. Обе системы — frozen-CLIP-encoder + MLP + LLM.

Ни одна не обрабатывает длинное видео. Обе являются системами на 8-16 кадров.

### Qwen2.5-VL and TMRoPE

Qwen2.5-VL ввела TMRoPE — Temporal-Modality Rotary Position Embedding. Каждый patch token несет позицию (t, h, w), где t — фактический timestamp (а не индекс кадра).

Ключевые отличия от простого temporal embedding:

- Абсолютное время, а не индекс. Модель видит "at 4.2 seconds", а не "at frame 15."
- Вращение на токен, а не на клип. Каждый визуальный токен вращается независимо по своему timestamp.
- Совместимость с dynamic FPS. Если здесь вы сэмплируете на 2 FPS, а там на 4 FPS, TMRoPE нативно обрабатывает неравномерные интервалы.

TMRoPE делает возможными запросы "at what second does the cat jump?". Модель может ответить "at 4.2 seconds." Video-LLaMA могла сказать только "early in the clip."

### Frame sampling strategies

Uniform: сэмплировать N кадров равномерно по длительности. Просто, но теряет пики движения.

Dynamic FPS: сэмплировать адаптивно на основе интенсивности движения. Optical flow или frame differencing выбирает сегменты с большим движением для более плотного сэмплинга. Qwen2.5-VL обучается на этом.

Event-driven: запускать легкий detector, сэмплировать больше там, где происходит действие. Используется VideoAgent.

Keyframe + context: сэмплировать на границах сцен + несколько соседних кадров. Используется для кинематографического контента.

### Pooling per frame

При 1 FPS и 576 токенах на кадр 5-минутный клип — это 172,800 токенов. Выполнимо с 128k context у Qwen2.5-VL-72B, но дорого.

3x3 bilinear pool сокращает до 64 токенов на кадр -> 19,200 токенов для 5 минут. Удачная точка для большинства задач.

Более агрессивный пулинг (6x6 -> 16 токенов на кадр) подходит для agent workflows, где пространственная детализация менее важна.

### The four video benchmarks

- VideoMME: комплексное понимание видео, short + medium + long.
- TempCompass: тонкое временное рассуждение, вопросы "before" / "after".
- EgoSchema: long-horizon видео от первого лица.
- Video-MMMU: мультимодальные мультидисциплинарные вопросы по видео.

Полная оценка video-VLM затрагивает все четыре. Они нагружают разные оси — TempCompass полностью про порядок, EgoSchema про рассуждение на 3+ минутах, VideoMME охватывает разные длительности.

### Grounding output formats

Форматы вывода для temporal grounding:

- Free text: "The cat jumps around the 4-second mark." Легко парсить, но неточно.
- Structured JSON: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL обучается этому.
- Token-based: специальные токены `<time>4.1</time>`, вставленные в ответ. Внутренний формат Qwen2.5-VL.

Token-based наиболее точен для downstream use. JSON-формат вывода Qwen2.5-VL парсится напрямую.

### 2026 best practice

Для video VLMs в 2026 году:

- Encoder: SigLIP 2 с M-RoPE или TMRoPE (Qwen2.5-VL).
- Frame sampling: dynamic FPS (1-4 в зависимости от движения) с max-frame cap.
- Per-frame pooling: 3x3 bilinear.
- Output: structured JSON с полями time + event.
- Benchmarks: VideoMME + TempCompass для общего качества; EgoSchema для long-horizon.

## Использование

`code/main.py` включает:

- Uniform и dynamic-FPS frame samplers.
- Игрушечный temporal-grounding evaluator: по "ground truth" событию во время T и model output оценивает точность с tolerance.
- Сравнение Video-LLaMA (16 frames, Q-former), Video-LLaVA (8 frames, MLP), Qwen2.5-VL (dynamic FPS + TMRoPE).

## Результат

Этот урок производит `outputs/skill-video-vlm-frame-planner.md`. По видео-задаче (monitoring, action recognition, temporal grounding, summarization) он выбирает frame sampler, pooling factor, output format и ожидаемый accuracy tier.

## Упражнения

1. Для 3-минутной кулинарной демонстрации выберите uniform или dynamic FPS. Обоснуйте количеством токенов.

2. Что именно добавляет TMRoPE, чего не может сделать простая temporal embedding table?

3. Напишите JSON schema для temporal grounding, которую VLM может научиться выдавать. Включите error cases.

4. Прочитайте Section 3 Video-LLaVA про "Alignment Before Projection." Почему это лучше, чем обучать отдельные image и video encoders?

5. По leaderboard VideoMME: каков разрыв между лучшей открытой моделью и лучшей проприетарной моделью по состоянию на 2026 год? Какая часть этого разрыва объясняется temporal encoding, а какая — масштабом base LLM?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Temporal grounding | "Time-localized answers" | VLM выдает конкретный диапазон timestamp, когда происходит событие |
| TMRoPE | "Time-Multimodal RoPE" | 3D rotary position с абсолютными timestamps, используется Qwen2.5-VL |
| Dynamic FPS | "Motion-aware sampling" | Сэмплировать больше кадров в сегментах с сильным движением и меньше в статичных |
| Frame pooling | "Spatial compress per frame" | Уменьшить число patches на кадр с помощью bilinear interpolation перед LLM |
| Video Q-former | "Clip compressor" | Cross-attention bottleneck, отображающий N кадров в K learned queries |
| VideoMME | "Video bench" | Комплексный short/medium/long video benchmark, 2500+ samples |

## Дополнительное чтение

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
