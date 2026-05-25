# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

> "Вежливо попросить модель вернуть JSON" ломается в 5-15 процентах случаев даже на frontier models. Structured outputs закрывают этот разрыв через constrained decoding: модели буквально запрещено выдавать token, который нарушил бы schema. OpenAI strict mode, schema-typed tool use Anthropic, `responseSchema` Gemini, `output_type` Pydantic AI и `.parse` Zod — пять поверхностных форм одной идеи. Этот урок строит schema validator и strict-mode contract, которые учащиеся будут использовать для каждого production extraction pipeline.

**Тип:** практика
**Языки:** Python (stdlib, подмножество JSON Schema 2020-12)
**Предварительные знания:** Phase 13 · 02 (глубокий разбор function calling)
**Время:** ~75 минут

## Цели обучения

- Написать JSON Schema 2020-12 для extraction target с правильными constraints (enum, min/max, required, pattern).
- Объяснить, почему strict mode и constrained decoding дают другие гарантии, чем "валидация после генерации".
- Различать три failure modes: parse error, schema violation, model refusal.
- Ship-ить extraction pipeline с typed repair и typed refusal handling.

## Проблема

Агент, читающий purchase-order email, должен превратить свободный текст в `{customer, line_items, total_usd}`. Есть три подхода.

**Подход первый: попросить JSON в prompt.** "Ответь JSON с полями customer, line_items, total_usd." Работает 85-95 процентов времени на frontier models. Ломается шестью способами: отсутствующая скобка, trailing comma, неправильные типы, сгаллюцинированные fields, обрезание на token limit, leaked prose вроде "Here is your JSON:".

**Подход второй: валидировать после генерации.** Сгенерировать свободно, распарсить, проверить по schema, сделать retry при failure. Надежно, но дорого: вы платите за каждый retry, а truncation bugs стоят одного extra turn на occurrence.

**Подход третий: constrained decoding.** Provider обеспечивает schema во время decoding. Невалидные tokens маскируются из sampling distribution. Output гарантированно парсится и гарантированно валидируется. Failure схлопывается в один mode: refusal (модель решает, что input не подходит под schema).

Каждый frontier provider 2026 года поставляет какую-то форму третьего подхода.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` плюс `refusal` в response, если модель отказывается.
- **Anthropic.** Schema enforcement на inputs `tool_use`; `stop_reason: "refusal"` не существует, но `end_turn` без tool call является signal.
- **Gemini.** `responseSchema` на request level; в 2026 Gemini ship-ит token-level grammar constraints для selected types.
- **Pydantic AI.** `output_type=InvoiceModel` выдает structured `RunResult`, типизированный как `InvoiceModel`.
- **Zod (TypeScript).** Runtime parser, который валидирует output provider по Zod schema; сочетается с `beta.chat.completions.parse` OpenAI.

Общая нить: объявить schema один раз и enforce ее end to end.

## Концепция

### JSON Schema 2020-12 — lingua franca

Каждый provider принимает JSON Schema 2020-12. Конструкции, которые вы используете чаще всего:

- `type`: одно из `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: map от имени field к subschema.
- `required`: list имен fields, которые обязаны присутствовать.
- `enum`: закрытый набор разрешенных значений.
- `minimum` / `maximum` (numbers), `minLength` / `maxLength` / `pattern` (strings).
- `items`: subschema, применяемая к каждому элементу array.
- `additionalProperties`: `false` запрещает лишние fields (default varies by mode).

OpenAI strict mode добавляет три требования: каждое property должно быть перечислено в `required`, `additionalProperties: false` должен стоять везде, и не должно быть unresolved `$ref`. Если нарушить их, API вернет 400 во время request.

### Pydantic, Python binding

Pydantic v2 генерирует JSON Schema из dataclass-shaped models через `model_json_schema()`. Pydantic AI оборачивает это, чтобы вы писали:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

а agent framework переводил schema в OpenAI strict mode, Anthropic `input_schema` или Gemini `responseSchema` на границе. Output модели возвращается как typed instance `Invoice`. Validation errors вызывают `ValidationError` с типизированными путями ошибок.

### Zod, TypeScript binding

Zod (`z.object({customer: z.string(), ...})`) — TS-эквивалент. Node SDK OpenAI предоставляет `zodResponseFormat(Invoice)`, который переводит это в JSON Schema payload API.

### Refusals

Strict mode не может заставить модель ответить. Если input не помещается в schema ("email был стихотворением, а не invoice"), модель выдает поле `refusal` с reason. Ваш код должен обрабатывать это как first-class outcome, а не failure. Refusal также полезен как safety signal: модель, которую попросили извлечь credit card number из protected-content email, возвращает refusal with safety reason attached.

### Constrained decoding в open

Open-weights implementations используют три техники.

1. **Grammar-based decoding** (`outlines`, `guidance`, `lm-format-enforcer`): строит deterministic finite automaton из schema; на каждом step маскирует logits tokens, которые нарушили бы FSM.
2. **Logit masking with a JSON parser**: запускает streaming JSON parser синхронно с model; на каждом step вычисляет valid-next-token set.
3. **Speculative decoding with a verifier**: дешевый draft model proposes tokens, verifier enforces schema.

Commercial providers выбирают одну из них behind the scenes. State of the art 2026 года быстрее plain generation для коротких structured outputs и примерно той же скорости для длинных.

### Три failure modes

1. **Parse error.** Output не является valid JSON. Не может случиться в strict mode. Все еще может случаться на non-strict providers.
2. **Schema violation.** Output парсится, но нарушает schema. Не может случиться в strict mode. Часто вне него.
3. **Refusal.** Модель отказывается. Должен обрабатываться как typed outcome.

### Retry strategy

Когда вы вне strict mode (Anthropic tool use, non-strict OpenAI, older Gemini), recovery pattern:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Одного retry обычно достаточно. Три retries ловят flakes слабых моделей. Больше трех — признак плохой schema: модель не может удовлетворить ее на некоторых inputs, и prompt или schema нужно чинить.

### Small-model support

Constrained decoding работает на small models. Open model на 3B parameters с grammar enforcement превосходит 70B-parameter model with raw prompting на structured tasks. Это главная причина, почему structured outputs важны для production: они отделяют reliability от model size.

## Используйте

`code/main.py` поставляет minimal JSON Schema 2020-12 validator на stdlib (types, required, enum, min/max, pattern, items, additionalProperties). Он оборачивает schema `Invoice` и прогоняет fake LLM output через validator, демонстрируя parse error, schema violation и refusal paths. В production замените fake output на real response любого provider.

На что смотреть:

- Validator возвращает typed list `[ValidationError]` с path и message. Именно эту форму нужно передавать в retry prompt.
- Refusal branch НЕ делает retry. Он логирует и возвращает typed refusal. Phase 14 · 09 использует refusals как safety signal.
- Проверка `additionalProperties: false` срабатывает на adversarial test input, показывая, почему strict mode закрывает дверь hallucinated fields.

## Отправьте

Этот урок создает `outputs/skill-structured-output-designer.md`. Получив free-text extraction target (invoices, support tickets, resumes, etc.), skill создает JSON Schema 2020-12, совместимую со strict mode, и Pydantic model, которая mirrors it, со stub-ами typed refusal и retry handling.

## Упражнения

1. Запустите `code/main.py`. Добавьте четвертый test case, где `total_usd` является negative number. Подтвердите, что validator отклоняет его с path constraint `minimum`.

2. Расширьте validator для поддержки `oneOf` with a discriminator. Common case: `line_item` либо product, либо service, tagged by `kind`. Strict mode имеет subtle rules; проверьте руководство OpenAI по structured outputs.

3. Напишите ту же Invoice schema как Pydantic BaseModel и сравните output `model_json_schema()` с hand-rolled schema. Найдите одно поле, которое Pydantic sets by default, а hand-rolled version omits.

4. Измерьте refusal rates. Сконструируйте десять inputs, которые не должны быть extractable (song lyric, math proof, blank email), и прогоните через real provider со strict mode. Посчитайте refusals vs hallucinated outputs. Это ваш ground truth для refusal-aware retries.

5. Прочитайте руководство OpenAI по structured outputs от начала до конца. Найдите одну construct, которую он явно forbids in strict mode, хотя plain JSON Schema allows. Затем спроектируйте schema, которая uses forbidden construct non-essentially, и refactor it to be strict-compatible.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "Спецификация schema" | IETF-draft schema dialect, на котором говорит каждый modern provider |
| Strict mode | "Гарантированная schema" | Флаг OpenAI, enforcing schema через constrained decoding |
| Constrained decoding | "Logit masking" | Decode-time enforcement, которое маскирует invalid next-tokens |
| Refusal | "Модель отказывается" | Typed outcome, когда input не может fit the schema |
| Parse error | "Invalid JSON" | Output не распарсился как JSON; impossible under strict |
| Schema violation | "Неправильная форма" | Parsed, но нарушил types / required / enum / range |
| `additionalProperties: false` | "Лишние поля запрещены" | Запрещает unknown fields; required in OpenAI strict |
| Pydantic BaseModel | "Typed output" | Python class, который emits and validates JSON Schema |
| Zod schema | "TypeScript output type" | TS runtime schema для provider output validation |
| Grammar enforcement | "Open-weights constrained decode" | FSM-based logit masking, как в outlines / guidance |

## Дополнительное чтение

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode, refusals и schema requirements
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — launch post августа 2024 года, объясняющий decoding guarantee
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — typed output_type bindings, которые сериализуются для каждого provider
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — каноническая spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — enterprise deployment notes и strict-mode caveats
