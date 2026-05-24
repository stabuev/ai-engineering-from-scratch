# Понимание документов и диаграмм

> Документы — это не фотографии. PDF, научная статья, счет или рукописная форма имеют layout, таблицы, диаграммы, сноски, заголовки и семантическую структуру, которую обычное понимание изображений не захватывает. Стек до VLM был конвейером: Tesseract OCR + LayoutLMv3 + эвристики извлечения таблиц. Волна VLM заменила это OCR-free моделями — Donut (2022), Nougat (2023), DocLLM (2023), — которые напрямую выдают структурированную разметку. К 2026 frontier — это просто "feed the page image to Claude Opus 4.7 at 2576px native," и structured-markup output получается бесплатно. Этот урок разбирает трехэпохальную дугу document AI.

**Тип:** Практика
**Языки:** Python (stdlib, скелет layout-aware document parser)
**Предварительные требования:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Время:** ~180 минут

## Цели обучения

- Объяснить три эпохи document AI: OCR pipeline, OCR-free, VLM-native.
- Описать три входных потока LayoutLMv3: text, layout (bbox), image patches, с unified masking.
- Сравнить Donut (OCR-free, image → markup), Nougat (scientific paper → LaTeX), DocLLM (layout-aware generative), PaliGemma 2 (VLM-native).
- Выбрать документную модель для новой задачи (invoices, scientific papers, handwritten forms, Chinese receipts).

## Проблема

"Понять этот PDF" обманчиво сложно. Информация находится в:

- Текстовом содержимом (90% сигнала).
- Layout (headers, footnotes, sidebars, two-column format).
- Таблицах (rows, columns, merged cells).
- Фигурах и диаграммах.
- Рукописных аннотациях.
- Шрифтах и типографике (title vs body).

Сырой OCR выгружает текст и теряет все остальное. Система, которой важны счета, должна знать, что "Total: $1,245" пришло из bottom-right, а не из footnote.

## Концепция

### Эпоха 1 — OCR pipeline (до 2021)

Классический стек:

1. PDF → image per page.
2. Tesseract (или коммерческий OCR) извлекает текст с per-word bounding boxes.
3. Layout analyzer определяет блоки (header, table, paragraph).
4. Table structure recognizer разбирает таблицы.
5. Domain rules + regex извлекают поля.

Работает для чистого печатного текста. Ломается на handwriting, skewed scans, complex tables, non-English scripts. Каждый режим отказа требует собственного exception path.

### TrOCR (2021)

TrOCR (Li et al., arXiv:2109.10282) заменил классический CNN-CTC в Tesseract на transformer encoder-decoder, обученный на synthetic + real text images. Чистая победа на рукописном и многоязычном тексте. Все еще pipeline (detector then TrOCR then layout), но шаг OCR резко улучшился.

### Эпоха 2 — OCR-free (2022-2023)

Первые OCR-free модели сказали: полностью пропустить detection, отображать пиксели изображения напрямую в структурированный выход.

Donut (Kim et al., arXiv:2111.15664):
- Encoder-decoder transformer, encoder is Swin-B.
- Выход — JSON для form understanding, markdown для summarization или любая task-specific schema.
- Нет OCR, нет layout, нет detection.

Nougat (Blecher et al., arXiv:2308.13418):
- Обучен специально на научных статьях.
- Выход — LaTeX / markdown.
- Обрабатывает equations, multi-column layout, figures.
- Модель, которую вызывает каждый arXiv-parser.

Это специалисты, а не универсалы. Donut на научной статье проваливается; Nougat на invoice проваливается.

### LayoutLMv3 (2022)

Другой путь. LayoutLMv3 (Huang et al., arXiv:2204.08387) сохраняет OCR, но добавляет понимание layout:

- Три входных потока: OCR text tokens, per-token 2D bounding boxes, image patches.
- Masked training objective по всем трем модальностям (masked text, masked patches, masked layout).
- Downstream: classification, entity extraction, table QA.

LayoutLMv3 — вершина OCR-based document understanding. Сильна на forms и invoices. Требует OCR upstream. Лучшая pre-VLM точность на стандартизованных document benchmarks.

### DocLLM (2023)

DocLLM (Wang et al., arXiv:2401.00908) — генеративный родственник LayoutLM. Генерирует свободные ответы, обусловленные layout tokens. Лучше для QA по документам; все еще зависит от OCR input.

### Эпоха 3 — VLM-native (2024+)

В 2024 VLM стали достаточно хороши, чтобы полностью заменить pipeline. Подайте полное изображение страницы в высоком разрешении VLM, задайте вопрос, получите ответ.

- LLaVA-NeXT 336-tile AnyRes работает для небольших документов.
- Qwen2.5-VL dynamic-resolution нативно обрабатывает 2048+ pixels.
- Claude Opus 4.7 поддерживает 2576px documents.
- PaliGemma 2 (апрель 2025) обучается специально для documents + handwriting.

Разрыв между VLM-native и OCR-pipeline быстро закрылся. К 2026 VLM-native побеждает на:

- Scene text (hand-written + printed, mixed scripts).
- Complex tables with merged cells.
- Math equations embedded in text.
- Figures with text annotations.

OCR pipelines все еще выигрывают на:

- Pure-scan workloads в огромном масштабе, где важна per-page latency.
- Надежности pipeline (детерминированные отказы vs VLM hallucinations).
- Регулируемых средах, требующих auditable OCR output.

### Frontier Claude 4.7 / GPT-5

При native input 2576 pixels frontier VLM выполняют document understanding с почти человеческой точностью. Бенчмарк-числа начала 2026:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, pipelined LayoutLMv3 ~83.
- ChartQA: Claude 4.7 ~92.2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

Разрыв closed-model в основном связан с разрешением и масштабом base-LLM. Открытые модели на 7B отстают на несколько пунктов, но догоняют.

### Математические уравнения и LaTeX output

Научным статьям нужен точный LaTeX output для уравнений. Nougat обучался на этом. VLM, обученные с LaTeX targets (Qwen2.5-VL-Math, derivatives Nougat), выдают пригодный LaTeX. Без явного LaTeX training VLM выдают читаемые, но неточные транскрипции.

Для scientific-paper pipelines в 2026: цепочка Nougat на PDF, затем VLM на сложных страницах.

### Рукописный текст

Все еще самая сложная подзадача. Mixed printed + handwritten (doctors' notes, filled forms) — место, где OCR pipelines все еще выигрывают у VLM по стоимости. Handwritten-only VLM улучшаются (Claude 4.7, PaliGemma 2).

### Рецепт 2026

Для нового document-AI project:

- Pure-printed invoices at scale: LayoutLMv3 + rules, cost-efficient.
- Mixed documents (scientific + handwritten + forms): VLM-native (PaliGemma 2 or Qwen2.5-VL).
- Full arXiv ingestion: Nougat for math, VLM for figures.
- Regulatory: OCR pipeline + VLM validator for cross-check.

## Применение

`code/main.py`:

- Игрушечный layout-aware tokenizer: по парам (text, bbox) создает вход в стиле LayoutLMv3.
- Генератор task schema в стиле Donut: JSON template for forms.
- Сравнение token budgets per page для OCR-pipeline, Donut, Nougat и VLM-native.

## Результат

Этот урок создает `outputs/skill-document-ai-stack-picker.md`. По document-AI project (domain, scale, quality, regulatory) выбирает между OCR pipeline, OCR-free specialist и VLM-native.

## Упражнения

1. Ваш проект обрабатывает 10M invoices per day. Какой стек минимизирует cost-per-page без потери accuracy?

2. Почему LayoutLMv3 превосходит pure-CLIP-VLMs на form QA, но уступает на scene-text? Что дает и что теряет bbox stream?

3. Nougat генерирует LaTeX. Предложите тестовый случай, где VLM-native output превосходит Nougat по LaTeX fidelity, и случай, где выигрывает Nougat.

4. Прочитайте статью PaliGemma 2 (Google, 2024). Какое ключевое добавление training-data подняло document accuracy по сравнению с PaliGemma 1?

5. Спроектируйте regulatory-safe hybrid: OCR pipeline как primary, VLM как secondary cross-check. Как вы разрешаете disagreement?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| OCR pipeline | "Tesseract-style" | Поэтапный стек: detect -> OCR -> layout -> rules; детерминированный, хрупкий |
| OCR-free | "Donut-style" | Image-to-output transformer, который пропускает явный OCR; единая модель |
| Layout-aware | "LayoutLM" | Вход включает per-token bbox coordinates; unified masking across modalities |
| VLM-native | "Frontier VLM" | Подать page image напрямую в Claude/GPT/Qwen VLM в high resolution; без pipeline |
| DocVQA | "Doc benchmark" | Стандарт document VQA; самая цитируемая метрика |
| Markup output | "LaTeX / MD" | Structured output format вместо free-form text; включает downstream automation |

## Дополнительное чтение

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
