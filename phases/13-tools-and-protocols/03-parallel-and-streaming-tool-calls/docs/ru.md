# Параллельные tool calls и streaming с инструментами

> Три независимых weather lookups, выполненные последовательно, — это три round trips. Запустите их параллельно, и общее время сожмется до самого медленного отдельного вызова. Теперь каждый frontier provider умеет выдавать несколько tool calls за один turn. Выигрыш реален; plumbing тонкий. Этот урок проходит обе половины: parallel fan-out и сборку streamed arguments, с акцентом на ловушку id-correlation.

**Тип:** практика
**Языки:** Python (stdlib, thread pool + streaming harness)
**Предварительные знания:** Phase 13 · 02 (глубокий разбор function calling)
**Время:** ~75 минут

## Цели обучения

- Объяснить, зачем существует `parallel_tool_calls: true` и когда его отключать.
- Сопоставлять chunks streamed arguments с правильным tool-call id во время parallel fan-out.
- Собирать частичные строки `arguments` в полный JSON без преждевременного парсинга.
- Запустить weather benchmark для трех городов, демонстрирующий sequential vs parallel latency.

## Проблема

Без parallel calls агент, отвечающий "какая погода в Бангалоре, Токио и Цюрихе", делает это:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Три LLM round trips, каждый из которых также платит executor latency. Примерно в 4 раза больше идеального wall-clock time.

С parallel calls:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Один LLM round trip. Executor time — максимум из трех, а не сумма. Production benchmarks на OpenAI, Anthropic и Gemini показывают 60-70 процентов сокращения wall-clock на fan-out workloads.

Цена — сложность корреляции. Когда три calls завершаются не по порядку, results должны нести соответствующий `tool_call_id`, чтобы модель могла их сопоставить. Когда results stream-ятся, нужно собрать частичные fragments arguments в полный JSON до execution. Gemini 3 добавила unique ids в том числе для решения реальной проблемы, где два parallel calls к одному и тому же tool были неотличимы.

## Концепция

### Включение parallel

- **OpenAI.** `parallel_tool_calls: true` включено по умолчанию. Установите `false`, чтобы принудительно сделать serial.
- **Anthropic.** Parallel через `disable_parallel_tool_use: false` (по умолчанию в Claude 3.5 и выше). Установите `true` для serial.
- **Gemini.** Всегда parallel-capable; `tool_config.function_calling_config.mode = "AUTO"` позволяет модели решать.

Отключайте parallel, когда у tools есть зависимости порядка (`create_file`, затем `write_file`), когда output одного call влияет на input другого или когда rate limiter не выдержит fan-out.

### Корреляция id

У каждого call, выданного моделью, есть `id`. Каждый result, возвращаемый host, должен включать тот же id. Без этого results неоднозначны.

- **OpenAI.** `tool_call_id` на каждом tool-role message.
- **Anthropic.** `tool_use_id` на каждом block `tool_result`.
- **Gemini.** `id` на каждом `functionResponse` (Gemini 3 и выше; Gemini 2 сопоставляла по name, что ломалось для same-name parallel calls).

### Параллельное выполнение calls

Host запускает executor каждого call в отдельном thread, coroutine или remote worker. Самый простой harness использует thread pool; production использует asyncio с `asyncio.gather` или structured concurrency. Порядок завершения непредсказуем — id является идентификатором.

Один частый bug: отвечать results в порядке call-list, а не в порядке завершения. Обычно это работает, потому что модели важен только `tool_call_id`, но если result потерян или продублирован, out-of-order submission усложняет debugging. Предпочитайте отвечать в completion order с явными ids.

### Streaming tool calls

Когда модель stream-ит, `arguments` приходят кусками. Chunks трех отдельных stream-ов для трех parallel calls перемежаются на wire. Нужен один accumulator на id.

Форма по provider:

- **OpenAI.** Каждый chunk — `choices[0].delta.tool_calls[i].function.arguments` (partial string). Chunk несет `index` (позиция в списке calls). Вы накапливаете per-index, читаете `id`, когда он впервые появляется, и парсите JSON при `finish_reason = "tool_calls"`.
- **Anthropic.** Stream events: `message_start`, затем по одному `content_block_start` на block с type `tool_use` (с id, name, empty input). События `content_block_delta` несут chunks `input_json_delta`. `content_block_stop` закрывает каждый block.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 и выше) выдает chunks с `functionCallId`, чтобы calls чисто перемежались. До Gemini 3 streaming возвращал один complete call за раз.

### Partial JSON и ловушка parse-early

Нельзя парсить `arguments`, пока они не завершены. Partial JSON вроде `{"city": "Beng` невалиден и вызовет exception. Правильный gate — provider-specific сигнал конца call: OpenAI `finish_reason = "tool_calls"`, Anthropic `content_block_stop` или Gemini stream-end event. Только тогда пытайтесь делать `json.loads`. Более robust approach использует incremental JSON parser, который выдает events по мере завершения структуры; OpenAI streaming guide рекомендует это для UX, показывающего live-индикатор "thinking". Подсчет скобок ненадежен как проверка полноты (скобки внутри quoted strings или escaped content дают false positives) и должен использоваться только как неформальная эвристика debugging.

### Out-of-order completion

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Reply от host все равно должен ссылаться на ids:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

Порядок в reply не важен для корректности в OpenAI или Anthropic. Gemini принимает любой order, пока ids match.

### Benchmark: sequential vs parallel

Harness в `code/main.py` симулирует три executors с latency 400, 600 и 800 ms. Sequential выполняет их за 1800 ms total. Parallel выполняет за max(400, 600, 800) = 800 ms. Разница постоянная, а не пропорциональная, поэтому savings растут с числом tools.

Практическая оговорка: parallel calls нагружают downstream APIs. 10-way fan-out к rate-limited service упадет. Phase 13 · 17 покрывает gateway-level backpressure; retry semantics запланированы для будущей phase.

### Streaming fan-out wall-clock

Если сама модель stream-ит, можно начать execution сразу, как только arguments одного call complete, а не ждать finalize всех calls. Это optimization, которую OpenAI документирует, но не все SDKs expose. Harness в этом уроке делает это: как только simulated stream выдает complete argument object, host запускает этот call.

## Используйте

`code/main.py` состоит из двух половин. Первая запускает три simulated weather calls sequentially и in parallel с помощью `concurrent.futures.ThreadPoolExecutor` и печатает wall-clock time. Вторая replay-ит fake streaming response — chunks `arguments` для трех parallel calls, перемежающиеся в одном stream, — и собирает их per-id через `StreamAccumulator`. Без LLM, без network, только логика сборки.

На что смотреть:

- Sequential timer показывает 1.8 seconds. Parallel timer показывает 0.8 seconds на тех же fake latencies.
- Accumulator обрабатывает chunks, пришедшие out of order, буферизует per-id и парсит только когда JSON каждого call complete.
- Executor запускается сразу после finalize arguments конкретного id, а не после окончания всех streams.

## Отправьте

Этот урок создает `outputs/skill-parallel-call-safety-check.md`. Получив tool registry, skill проверяет, какие tools безопасно запускать параллельно, какие имеют зависимости порядка и какие перегрузят downstream rate limits, возвращая revised registry с флагами `parallel_safe` для каждого tool.

## Упражнения

1. Запустите `code/main.py` и меняйте simulated latencies. Подтвердите, что parallel-to-sequential ratio примерно равен `max/sum` (real runs слегка отклоняются от идеала из-за thread scheduling, serialization и overhead harness). При каком latency distribution parallel перестает иметь значение?

2. Расширьте accumulator, чтобы он обрабатывал случай "call был отменен mid-stream": сбрасывал его buffer и выдавал event `cancelled`. Какой provider явно документирует этот случай? Проверьте semantics Anthropic `content_block_stop` и behavior OpenAI `finish_reason: "length"`.

3. Замените thread pool на `asyncio.gather`. Проведите benchmark обоих вариантов. Вы должны увидеть небольшие wins на async из-за меньшей context-switch cost, но только если executors делают настоящий I/O.

4. Выберите два tools, которые НЕ должны запускаться параллельно (например, `create_file`, затем `write_file`). Добавьте graph `ordering_dependency` в registry и поставьте gate для parallel fan-out на основе этого graph. Это минимальный механизм dependency-aware scheduling, который формализует будущая phase по agent engineering.

5. Прочитайте раздел OpenAI parallel-function-calling и docs Anthropic `disable_parallel_tool_use`. Найдите один real-world tool type, для которого Anthropic рекомендует disabling parallelism. (Подсказка: consequential mutations on the same resource.)

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Parallel tool calls | "Fan-out за один turn" | Модель выдает несколько tool calls в одном assistant message |
| `parallel_tool_calls` | "Флаг OpenAI" | Включает или отключает выдачу нескольких calls |
| `disable_parallel_tool_use` | "Обратный флаг Anthropic" | Opt-out flag; по умолчанию parallel включен |
| Tool call id | "Correlation handle" | Идентификатор per-call, который result message должен повторить |
| Accumulator | "Stream buffer" | Per-id string buffer для partial chunks `arguments` |
| Out-of-order completion | "Самый быстрый первым" | Parallel calls завершаются в непредсказуемом порядке; ids связывают данные |
| Dependency graph | "Ограничения порядка" | Tools, outputs которых идут во inputs других tools; нельзя parallelize |
| Parse-early trap | "JSON.parse упал" | Попытка parse incomplete string `arguments` |
| `streamFunctionCallArguments` | "Фича Gemini 3" | Streamed argument chunks с unique id per call |
| Completion-order reply | "Не ждать всех" | Reply with results as they arrive, keyed by id |

## Дополнительное чтение

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — default behavior и opt-out flag
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` и result batching
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — id-correlated parallel calls в Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — chunked argument reassembly для OpenAI streams
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` с `input_json_delta`
