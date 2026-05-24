# Function Calling Deep Dive — OpenAI, Anthropic, Gemini

> Три frontier providers в 2024 году сошлись на одном и том же цикле tool-call, а затем разошлись во всем остальном. OpenAI использует `tools` и `tool_calls`. Anthropic использует блоки `tool_use` и `tool_result`. Gemini использует `functionDeclarations` и корреляцию по unique-id. Этот урок сравнивает все три варианта рядом, чтобы код, который отправляется на одном provider, не ломался при переносе на другой.

**Тип:** практика
**Языки:** Python (stdlib, schema translators)
**Предварительные знания:** Phase 13 · 01 (интерфейс инструментов)
**Время:** ~75 минут

## Цели обучения

- Назвать три различия формы между function-calling payloads OpenAI, Anthropic и Gemini (declaration, call, result).
- Перевести одно объявление инструмента во все три формата provider и предсказать, где strict-mode constraints будут различаться.
- Использовать `tool_choice` у каждого provider, чтобы принудительно выбрать, запретить или автоматически подобрать tool calls.
- Знать жесткие лимиты каждого provider (число инструментов, глубина schema, длина arguments) и signatures ошибок, которые они выдают при нарушении лимитов.

## Проблема

Форма function-calling request различается у разных providers. Три конкретных примера из production-стеков 2026 года:

**OpenAI Chat Completions / Responses API.** Вы передаете `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Ответ модели содержит `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`, где `arguments` — JSON-строка, которую вы должны распарсить. Strict mode (`strict: true`) обеспечивает schema compliance через constrained decoding.

**Anthropic Messages API.** Вы передаете `tools: [{name, description, input_schema}]`. Ответ приходит как `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` уже распарсен (объект, а не строка). Вы отвечаете новым сообщением `user`, содержащим блок `{type: "tool_result", tool_use_id, content}`.

**Google Gemini API.** Вы передаете `tools: [{functionDeclarations: [{name, description, parameters}]}]` (вложено под `functionDeclarations`). Ответ приходит как `candidates[0].content.parts: [{functionCall: {name, args, id}}]`, где `id` уникален в Gemini 3 и выше для correlation параллельных вызовов. Вы отвечаете `{functionResponse: {name, id, response}}`.

Один и тот же цикл. Разные имена полей, разная вложенность, разные соглашения string-vs-object, разные механизмы корреляции. Команда, написавшая weather agent на OpenAI, платит два дня за перенос plumbing на Anthropic и еще день на Gemini.

Этот урок строит translator, который сводит три формата к одному canonical tool declaration и маршрутизирует на границе. Phase 13 · 17 обобщает тот же паттерн в LLM gateway.

## Концепция

### Общая структура

Каждому provider нужны пять вещей:

1. **Tool list.** Name, description и input schema для каждого tool.
2. **Tool choice.** Force конкретный tool, forbid tools или дать модели решать.
3. **Call emission.** Structured output с именем tool и arguments.
4. **Call id.** Сопоставить response с правильным call (важно для parallel).
5. **Result injection.** Message или block, который связывает result обратно с call.

### Различия формы, поле за полем

| Аспект | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Envelope объявления | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Поле schema | `parameters` | `input_schema` | `parameters` |
| Контейнер response | `tool_calls[]` в assistant message | `content[]` с типом `tool_use` | `parts[]` с типом `functionCall` |
| Тип arguments | JSON как строка | распарсенный объект | распарсенный объект |
| Формат id | `call_...` (генерирует OpenAI) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Блок result | role `tool`, `tool_call_id` | `user` с `tool_result`, `tool_use_id` | `functionResponse` с matching `id` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema (always enforced) | `responseSchema` at request level |

### Лимиты, в которые вы реально упретесь

- **OpenAI.** 128 tools на request. Глубина schema 5. Строка arguments <= 8192 bytes. Strict mode требует: без `$ref`, без `oneOf`/`anyOf`/`allOf` с пересечениями, каждое property перечислено в `required`.
- **Anthropic.** 64 tools на request. Глубина schema формально почти не ограничена, но практический лимит 10. Нет strict-mode flag; schema является contract, и модель обычно ему следует.
- **Gemini.** 64 functions на request. Типы schema — подмножество OpenAPI 3.0 (небольшое расхождение с JSON Schema 2020-12). Unique-id для parallel calls начиная с Gemini 3.

### Поведение `tool_choice`

Три режима поддерживают все, но называют по-разному.

- **Auto.** Модель выбирает tool или text. Режим по умолчанию.
- **Required / Any.** Модель должна вызвать хотя бы один tool.
- **None.** Модель не должна вызывать tools.

Плюс по одному уникальному режиму у каждого provider:

- **OpenAI.** Force конкретный tool по name.
- **Anthropic.** Force конкретный tool по name; флаг `disable_parallel_tool_use` отделяет single от multi.
- **Gemini.** `mode: "VALIDATED"` прогоняет каждый response через schema validator независимо от намерения модели.

### Параллельные вызовы

OpenAI `parallel_tool_calls: true` (по умолчанию) выдает несколько calls в одном assistant message. Вы запускаете их все и отвечаете batched tool-role message с одной записью на каждый `tool_call_id`. Anthropic исторически делала single-call; `disable_parallel_tool_use: false` (по умолчанию начиная с Claude 3.5) включает multi. Gemini 2 позволяла parallel calls, но не давала stable ids; Gemini 3 добавляет UUIDs, чтобы out-of-order responses корректно сопоставлялись.

### Streaming

Все три поддерживают streamed tool calls. Wire format различается:

- **OpenAI.** Delta chunks `tool_calls[i].function.arguments` приходят постепенно. Вы накапливаете их до `finish_reason: "tool_calls"`.
- **Anthropic.** События block-start / block-delta / block-stop. Chunks `input_json_delta` несут partial arguments.
- **Gemini.** `streamFunctionCallArguments` (новое в Gemini 3) выдает chunks с `functionCallId`, чтобы несколько parallel calls могли перемежаться.

Phase 13 · 03 подробно разбирает parallel + streaming reassembly. Этот урок сосредоточен на declaration и single-call shapes.

### Ошибки и исправление

Ошибки invalid-argument тоже выглядят по-разному.

- **OpenAI (non-strict).** Модель возвращает `arguments: "{bad json}"`, парсинг JSON падает, вы добавляете error message и вызываете заново.
- **OpenAI (strict).** Validation происходит во время decoding; invalid JSON невозможен, но может появиться `refusal`.
- **Anthropic.** `input` может содержать неожиданные поля; schema advisory. Валидируйте на стороне server.
- **Gemini.** Особенность OpenAPI 3.0: `enum` на object fields может silently ignored; валидируйте сами.

### Паттерн translator

Canonical tool declaration в вашем коде выглядит так (форму выбираете вы):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Три маленькие функции переводят его в три provider shapes. Harness в `code/main.py` делает именно это, затем round-trip-ит fake tool call через response shape каждого provider. Network не нужен — этот урок учит формам, а не HTTP.

Production-команды заворачивают такой translator в `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph) или `BaseTool` (LlamaIndex). Phase 13 · 17 ship-ит gateway, который предоставляет OpenAI-shaped API поверх любого из трех providers.

## Используйте

`code/main.py` определяет один canonical dataclass `Tool` и три translators, которые выдают declaration JSON для OpenAI, Anthropic и Gemini. Затем он парсит вручную созданный provider response каждой формы в один и тот же canonical call object, показывая, что semantics под поверхностью идентичны. Запустите его и сравните три declarations рядом.

На что смотреть:

- Три declaration blocks отличаются только envelope и именами fields.
- Три response blocks отличаются тем, где живет call (top-level `tool_calls`, block `content[]`, entry `parts[]`).
- Одна функция `canonical_call()` извлекает `{id, name, args}` из всех трех response shapes.

## Отправьте

Этот урок создает `outputs/skill-provider-portability-audit.md`. Получив function-calling integration под одного provider, skill делает portability audit: на какие лимиты provider он опирается, какие fields нужно переименовать и что сломается при переносе на каждого другого provider.

## Exercises

1. Запустите `code/main.py` и проверьте, что все три provider declaration JSONs сериализуют один и тот же underlying объект `Tool`. Измените canonical tool, добавив enum parameter, и подтвердите, что только Gemini translator должен обработать OpenAPI quirk.

2. Добавьте parser `ListToolsResponse` для каждого provider, который извлекает tool list, возвращаемый моделью после `list_tools` или discovery call. У OpenAI такого нативно нет; отметьте эту асимметрию.

3. Реализуйте conversion для `tool_choice`: отобразите canonical `ToolChoice(mode="force", tool_name="x")` во все три provider shapes. Затем отобразите `mode="any"` и `mode="none"`. Проверьте diff table урока.

4. Выберите одного из трех providers и прочитайте его руководство по function calling от начала до конца. Найдите одно поле в его schema spec, которое два других не поддерживают. Кандидаты: OpenAI `strict`, Anthropic `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. Напишите test vector: tool call, чьи arguments нарушают declared schema. Прогоните его через validator каждого provider (stdlib-валидатор из Lesson 01 подойдет как proxy) и запишите, какие errors срабатывают. Документируйте, какого provider вы бы выбрали для production с точки зрения strictness.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Function calling | "Использование инструментов" | Provider-level API для выдачи structured tool-call |
| Tool declaration | "Спецификация инструмента" | Name + description + JSON Schema input payload |
| `tool_choice` | "Принудить / запретить" | Режимы Auto / required / none / specific-name |
| Strict mode | "Принудительная schema" | Флаг OpenAI, который ограничивает decoding соответствием schema |
| `tool_use` block | "Форма вызова Anthropic" | Inline content block с id, name, input |
| `functionCall` part | "Форма вызова Gemini" | Entry `parts[]`, содержащая name, args и id |
| Arguments-as-string | "JSON в строке" | OpenAI возвращает args как JSON string, а не object |
| Parallel tool calls | "Fan-out за один turn" | Несколько tool calls в одном assistant message |
| Refusal | "Модель отказывается" | Strict-mode-only refusal block вместо call |
| OpenAPI 3.0 subset | "Особенность schema в Gemini" | Gemini использует JSON-Schema-like dialect с небольшими отличиями |

## Дополнительное чтение

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — канонический справочник, включая strict mode и parallel calls
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — semantics блоков `tool_use` и `tool_result`
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — parallel calls, unique ids и OpenAPI subset
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — enterprise surface Gemini
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — детали strict-mode schema enforcement
