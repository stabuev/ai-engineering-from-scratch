# Batch APIs — скидка 50% как отраслевой стандарт

> Каждый крупный провайдер предоставляет асинхронный batch API со скидкой 50% и временем выполнения около 24 часов. OpenAI, Anthropic, Google и большинство inference-платформ (Fireworks batch tier, Together batch) реализуют один и тот же паттерн. Совместите batch с prompt caching, и ночные пайплайны опустятся примерно до 10% стоимости синхронного запуска без кэша. Правило предельно простое: если задача не интерактивная, ей место в batch. Пайплайны генерации контента, классификация документов, извлечение данных, генерация отчетов, массовая разметка, тегирование каталогов — все, что допускает задержку до 24 часов, оставляет деньги на столе, пока не переведено в batch. Производственный паттерн 2026 года — распределять каждую новую LLM-нагрузку по трем полосам: interactive (синхронно с кэшированием), semi-interactive (асинхронная очередь с fallback), batch (ночной запуск, с кэшированным входом). Нагрузки, которые притворяются интерактивными, но допускают задержку в минуты, тратят больше всего.

**Тип:** Learn
**Языки:** Python (stdlib, toy batch-vs-sync cost simulator)
**Предварительные требования:** Phase 17 · 14 (Prompt & Semantic Caching)
**Время:** ~45 minutes

## Цели обучения

- Назвать три provider batch API (OpenAI, Anthropic, Google) и общие гарантии: скидка 50% + выполнение за 24h.
- Посчитать стоимость комбинации batch + cached-input для ночной классификационной нагрузки и сравнить с синхронным baseline без кэша.
- Отнести workload к interactive / semi-interactive / batch и обосновать выбор полосы.
- Назвать две ловушки: partial interactivity (пользователь ожидает быстрее, чем 24h) и output-schema drift (batch file format различается у провайдеров).

## Проблема

Ваша команда поставляет ночной пайплайн генерации отчетов. 50 000 документов: кратко пересказать каждый, кластеризовать summary, подготовить executive brief. При синхронном запуске это занимает 4 часа и стоит $2,000 за ночь. Вы узнаете про batch APIs.

Batch дает скидку 50%. Вы также включаете prompt caching для system prompt (общего для всех 50k вызовов). В сумме счет падает до $180 за ночь — примерно 9% от baseline. Тот же пайплайн, три изменения конфигурации.

Batch — самый дешевый рычаг в наборе инструментов снижения LLM-стоимости, которым почти никто не пользуется. Причина в основном организационная: команды думают "real-time", хотя фактический SLA — "к утру". Этот урок о том, как не оставлять 90% счета на столе.

## Концепция

### The three batch APIs

**OpenAI Batch API**: загрузка JSONL-файла со списком запросов. Обещанный turnaround — 24 часа (на практике обычно ~2-8 часов). Скидка 50% на input и output tokens. Endpoint `/v1/batches`. Cache-eligible inputs также получают cached-input pricing сверху.

**Anthropic Message Batches**: загрузка JSONL. Turnaround 24 часа. Скидка 50%. Поддерживает `cache_control` — записи в кэш явные, чтения внутри batch происходят автоматически.

**Google Vertex AI Batch Prediction**: вход из BigQuery или GCS. Похожая скидка 50% для Gemini. Интегрируется с Vertex pipelines.

### Semantic: asynchronous, not slow

Batch означает "я обещаю вернуть результат в течение 24 часов", а не "это займет 24 часа". Типичный P50 — 2-6 часов. Провайдер планирует ваш batch на непиковые окна, когда GPU-инвентарь недоиспользован.

### Stack with caching

Summarization 50k документов с одинаковым system prompt на 4K токенов:

- Synchronous uncached: 50000 × ($input × 4000 + $output × 200) по полным тарифам.
- Synchronous cached: system prompt кэшируется после первой записи; оставшиеся 49999 получают input в 10x дешевле.
- Batch cached: все выше плюс скидка 50% и на read, и на write.

Комбинация: batch + cache = примерно 10% от sync uncached bill. Любая нагрузка, которая выполняется ночью и имеет общий system prompt, должна использовать это.

### Workload triage

**Interactive** — пользователь ждет ответ. TTFT важен. Синхронный вызов с prompt caching. Batch нельзя.

**Semi-interactive** — пользователь отправляет задачу и возвращается через минуты. Асинхронная очередь с fallback на sync, если batch недоступен. Пример: RAG indexing среднего объема.

**Batch** — пользователь ожидает результат "к утру" или "через час". Content pipelines, classification at scale, offline analysis. Всегда batch, всегда вместе с caching.

Частая ошибка: считать все interactive, потому что пайплайн production. Production — это не latency spec; важен SLA.

### The partial-interactivity trap

Некоторые функции выглядят интерактивными, но допускают 5-10 минут. Пример: ночной отчет о здоровье клиента с кнопкой "refresh". Пользователь нажимает refresh; подождать 10 минут нормально. Команда делает синхронную реализацию. 50 одновременных refresh стоят в 10x дороже, чем batched-and-delivered-via-email.

Вопрос, который нужно задать: "Что значит 24 часа для этого пользователя?" Если ответ "он бы не заметил", отправляйте в batch.

### The output-schema trap

Форматы batch-файлов различаются по провайдерам:

- OpenAI: JSONL, один request на строку.
- Anthropic: JSONL, одно message на строку; response format встроен.
- Vertex: BigQuery table или GCS prefix с TFRecord.

Написать "one batch client" для разных провайдеров означает adapter code для каждого провайдера. Gateways, которые заявляют multi-provider batch (Portkey, LiteLLM some tiers), все равно тонко оборачивают raw format.

### Numbers you should remember

- Batch discount across providers: фиксированные 50% на input + output.
- Turnaround SLA: гарантия 24 часа, типичный P50 — 2-6 часов.
- Stacked batch + cached input: ~10% стоимости sync uncached.
- Workload triage rule: если задержка 24h приемлема, всегда batch.

## Используйте это

`code/main.py` считает стоимость для sync, sync+cache, batch и batch+cache на workload из 50k документов. Показывает экономию в $ и процентах.

## Доведите до результата

Этот урок создает `outputs/skill-batch-triager.md`. По характеристикам workload он относит его к interactive/semi/batch и оценивает экономию.

## Упражнения

1. Запустите `code/main.py`. Для пайплайна на 100k документов с system prompt на 3K токенов и output на 500 токенов посчитайте экономию полного стека (batch + cache) относительно sync baseline.
2. Выберите три функции в реальном продукте, который вы знаете. Отнесите каждую к interactive/semi/batch.
3. Пользователь жалуется, что отчет занял 3 часа. Это batch mis-triage или действительно interactive? Запишите критерий решения.
4. SLA возврата вашего batch API — 24h, но P99 равен 20 часам. Как вы сообщаете это пользователю — как ведет себя downstream-система в edge case?
5. Посчитайте break-even: при какой длине shared-prefix batch + cache становится дешевле, чем ночной запуск на собственном reserved GPU?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Batch API | "async discount" | скидка 50% с turnaround 24h |
| JSONL | "batch format" | один JSON request на строку; стандарт OpenAI/Anthropic |
| Message Batches | "Anthropic batch" | название batch API продукта Anthropic |
| Batch prediction | "Vertex batch" | batch API продукт Vertex AI |
| Turnaround SLA | "24h promise" | гарантия, не типичное значение; обычно 2-6h |
| Workload triage | "interactivity decision" | решение о маршрутизации Interactive / semi / batch |
| Output schema | "response format" | JSONL layout на стороне провайдера; непереносим |
| Stacked discount | "batch + cache" | ~10% от uncached sync bill, когда применимы оба |

## Дополнительное чтение

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format and `/v1/batches` semantics.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format and `cache_control` interaction.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
