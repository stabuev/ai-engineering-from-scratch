# Понимание длинного видео при миллионном контексте

> 1-часовое 4K-видео при 24 FPS, разбитое на patches и embedded, дает порядка 60 миллионов токенов. Транскрипт 2-часового эпизода подкаста — 30,000 токенов. Полнометражный фильм Blu-ray, даже сжатый агрессивным пулингом, — это сотни тысяч токенов. Gemini 1.5 от Google (март 2024) открыла эту эпоху контекстом на 10 миллионов токенов, надежно выполняя needle-in-a-haystack recall по часовым видео. LWM (Liu et al., февраль 2024) показала путь масштабирования ring attention. LongVILA и Video-XL еще сильнее масштабировали ingestion. VideoAgent заменил сырой контекст агентным retrieval. Каждый подход — отдельный компромисс по compute, recall и инженерной сложности. В этом уроке они разбираются рядом.

**Тип:** Практика
**Языки:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Предварительные требования:** Phase 12 · 17 (video temporal tokens)
**Время:** ~180 минут

## Цели обучения

- Вычислять общее количество visual-token для длинного видео при разных FPS и pooling.
- Объяснить три пути масштабирования: brute context (Gemini 1.5), ring attention (LWM), token compression (LongVILA / Video-XL).
- Сравнить raw-context video VLMs и agentic-retrieval video VLMs (VideoAgent) по точности и задержке.
- Спроектировать needle-in-a-haystack test для 30-минутного видео и измерить recall на конкретной минуте.

## Проблема

Один кадр с patches размера Qwen2.5-VL при native resolution 384 — это ~729 токенов. При 3x3 pooling это 81 токен на кадр. 30-минутный клип при 1 FPS = 1800 кадров = 145,800 токенов. Выполнимо для открытых VLM 2025 года, но впритык. При 2 FPS — 291,600 токенов, что помещается только в самые большие контексты.

2-часовой фильм при 1 FPS — это 583k токенов. За пределами большинства открытых моделей 2026 года; требуется Gemini 2.5 Pro или более агрессивный pooling.

Появились три пути масштабирования.

## Концепция

### Path 1: Brute context (Gemini 1.5, Claude Opus)

Бросить hardware на проблему. Масштабировать context до миллионов токенов, обработать все за один forward pass.

Gemini 1.5 Pro вышла с 1M токенов; Gemini 1.5 Ultra — до 10M; Gemini 2.5 Pro в 2026 году надежно обрабатывает часы видео. Статья (arXiv:2403.05530) документирует needle-in-a-haystack recall на уровне 99.7% вплоть до ~9.5M токенов.

Инженерия: кастомная реализация attention с memory hierarchy (local + global + sparse) плюс MoE expert routing для эффективности long-context. Полные детали не опубликованы. Не open-source.

### Path 2: Ring attention (LWM, LongVILA)

Ring attention распределяет длинные последовательности по устройствам в "ring", где каждое устройство хранит chunk. Attention по полной последовательности выполняется так: каждое устройство отправляет свой chunk следующему по кольцу, вычисляет partial attention и агрегирует.

LWM (Liu et al., 2024) обучила таким способом модель с 1M-token context. Training compute масштабируется линейно с context, а не квадратично — квадратичный удар attention амортизируется по устройствам ring.

LongVILA (arXiv:2408.10188) адаптировала паттерн к VLM. 1400-frame videos при 192 токенах на кадр = 268k context, обучение с ring attention на 8-way parallelism.

### Path 3: Token compression (Video-XL, LongVA)

Дешевле, чем brute context: агрессивно сжимать до того, как LLM увидит последовательность.

Video-XL (arXiv:2409.14485) использует visual summary token: каждый клип из N кадров производит один "summary" token, который attends по этим N. На инференсе LLM видит один summary token на клип, резко уменьшая context.

LongVA расширяет context LLM с 200k до 2M с помощью техники "long context transfer". Обучение на long-context text переносится на long-context video через общее представление.

Token compression обменивает recall на конкретных timestamps на масштабируемость. Модель в целом знает, что произошло, но иногда пропускает точные кадры.

### Path 4: Agentic retrieval (VideoAgent)

Не подавать полное видео в LLM. Вместо этого трактовать видео как базу данных и использовать LLM для запросов к ней.

VideoAgent (arXiv:2403.10517):

1. LLM читает вопрос.
2. LLM просит retrieval tool найти релевантные клипы ("show me segments with a cat").
3. Tool возвращает timestamps подходящих клипов.
4. LLM читает эти клипы через VLM.
5. LLM составляет ответ или задает follow-up queries.

Это паттерн LLM-as-agent, примененный к длинному видео. Более дешевый inference (кодируются только релевантные клипы), более сложная инженерия (качество retrieval становится bottleneck).

### Needle-in-a-haystack benchmarks

Стандартный long-context test: вставить уникальный визуальный или текстовый маркер в случайную точку видео, затем задать запрос, требующий его вспомнить.

Метрика: Recall@k по длине видео и позиции маркера.

Gemini 2.5 Pro набирает >99% recall на видео до 90 минут. Открытые модели 72B (Qwen2.5-VL-72B, InternVL3-78B) набирают ~85-90% на 30 минутах и деградируют после 60.

VideoAgent может сравняться с raw-context моделями или превзойти их на 2+ часах, потому что retrieval попадает в needle, если tool хорош.

### Which path to pick

Для 15-минутного клипа с frontier accuracy: открытая 72B + native context обычно работает. Выбирайте Qwen2.5-VL-72B.

Для контента от 30 минут до 1 часа: LongVILA или Video-XL для open; Gemini 2.5 Pro для closed. Важна планка качества — frontier уходит в closed.

Для контента 2+ часа: VideoAgent или похожие retrieval patterns. Альтернатива — суммаризировать в меньшие chunks и подавать hierarchical summaries.

### 2026 production pattern

На практике production-пайплайны для длинного видео гибридные:

1. Запустить dynamic-FPS sampling + aggressive pooling по всему видео (получить 100k-token global representation).
2. Передать в 72B VLM для global summary.
3. Если пользователь задает детальные вопросы, запустить agentic retrieval, используя summary как index.

Это сочетает brute-context для общего понимания и retrieval для локальных деталей.

## Использование

`code/main.py`:

- Вычисляет token budgets для видео от 1 минуты до 3 часов при разных FPS + pooling.
- Симулирует needle-in-a-haystack run: вставляет marker в случайный timestamp, задает вопрос, оценивает recall.
- Включает simulator agentic-retrieval router, который выбирает конкретные клипы для подачи downstream VLM.

Запустите budget table и почувствуйте разрыв в масштабе.

## Результат

Этот урок производит `outputs/skill-long-video-strategy-planner.md`. По длительности видео и сложности запроса он выбирает между brute-context, compression и agentic retrieval, а также вычисляет latency + quality expectations.

## Упражнения

1. 45-минутная лекция при 1 FPS, 81 токен на кадр. Сколько токенов всего? В контексты каких моделей помещается?

2. Спроектируйте needle-in-a-haystack test: на какой минуте вы вставляете marker и какой точный формат запроса?

3. Сравните brute-context Qwen2.5-VL-72B (80k context) с VideoAgent (Claude 3.5 + retrieval) на 1-часовом видео. Кто выигрывает по recall? Кто выигрывает по latency?

4. Memory cost ring attention масштабируется линейно по sequence length и линейно по device count. Объясните почему и что ломается, если убрать ring-rotation phase.

5. Прочитайте Gemini 1.5 Section 5 про needle-in-a-haystack. Что статья обнаружила о recall на границе 1M и 10M токенов?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Brute context | "Just more tokens" | Масштабировать context LLM до миллионов токенов; обработать все за один проход |
| Ring attention | "LWM-style parallel" | Распределенный паттерн attention, где каждое устройство хранит chunk и вращает его |
| Token compression | "Summary tokens" | Уменьшить число токенов на клип через обучаемый compressor перед LLM |
| Needle-in-haystack | "NIH test" | Вставить уникальный marker в случайную точку и попросить модель вспомнить его на test time |
| Agentic retrieval | "LLM as query planner" | LLM просит retrieval tool найти релевантные клипы, читает их через VLM и составляет ответ |
| VideoAgent | "Retrieval pattern for video" | Каноничный agentic-retrieval дизайн: question -> tool -> clip -> answer |

## Дополнительное чтение

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
