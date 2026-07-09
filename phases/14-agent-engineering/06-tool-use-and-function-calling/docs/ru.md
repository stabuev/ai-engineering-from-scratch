# Tool Use и Function Calling

> Toolformer (Schick et al., 2023) начал self-supervised tool annotation. Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) задает планку 2026 года: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination. Single-turn решен. Memory, dynamic decision-making и long-horizon tool chains — нет.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 13 · 01 (Function Calling Deep Dive)
**Время:** ~60 минут

## Цели обучения

- Объяснить self-supervised training signal Toolformer: сохранять tool annotations только когда выполнение снижает next-token loss.
- Назвать пять evaluation categories BFCL V4 и что каждая измеряет.
- Реализовать tool registry на stdlib с валидацией schema, приведением argument и sandboxing выполнения.
- Диагностировать три открытые проблемы 2026 года: long-horizon tool chaining, dynamic decision-making и memory.

## Проблема

Ранний tool use спрашивал: может ли модель предсказать правильный function call? Современный tool use спрашивает: может ли модель связывать tools в цепочки на 40 шагов, с memory, с частичной наблюдаемостью, с восстановлением после сбоев инструментов и без галлюцинации несуществующих tools?

Toolformer установил baseline: модели могут учиться, когда вызывать tools, через self-supervision. BFCL V4 определяет evaluation target 2026 года. Промежуток между ними — пространство, в котором живут production agents.

## Концепция

### Toolformer (Schick et al., NeurIPS 2023)

Идея: позволить модели аннотировать собственный pretraining corpus кандидатами API calls. Для каждого кандидата — выполнить его. Сохранить annotation только если включение результата tool снижает loss на next token. Fine-tune на отфильтрованном corpus.

Покрытые tools: calculator, QA system, search engines, translator, calendar. Self-supervision signal полностью о том, помогает ли tool предсказывать текст — без human labels.

Результат масштабирования: tool use появляется на масштабе. Smaller models страдают от tool annotations; larger models выигрывают. Поэтому frontier models 2026 года имеют сильный tool use baked in, а большинству 7B models нужен явный tool-use fine-tuning, чтобы быть надежными.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL — de facto evaluation 2026 года. Состав V4:

- **Agentic (40%)** — полные траектории агентов: memory, multi-turn, dynamic decisions.
- **Multi-Turn (30%)** — интерактивные диалоги с цепочками tools.
- **Live (10%)** — реальные prompts от пользователей (более сложное распределение).
- **Non-Live (10%)** — синтетические тестовые случаи.
- **Hallucination (10%)** — определить, когда не нужно вызывать ни один tool.

V3 ввел state-based evaluation: после tool sequence проверять фактическое состояние API (например, "is the file created?"), а не match AST tool calls. V4 добавил категории web search, memory и format sensitivity.

Ключевой finding 2026 года: single-turn function calling почти решен. Сбои концентрируются в memory (перенос контекста между turns), dynamic decision-making (выбор tools на основе предыдущих результатов), long-horizon chains (drift after 20+ steps) и hallucination detection (отказ от вызова, когда не подходит ни один tool).

### Схема инструмента

У каждого provider есть schema. Они различаются в деталях, но имеют одну форму:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic использует `input_schema` напрямую. OpenAI использует `function.parameters`. Оба принимают JSON Schema. Descriptions are load-bearing — модель читает их, чтобы выбрать правильный tool. Плохие tool descriptions — root cause №1 для failures с неправильным выбором tool.

### Валидация аргументов

Не доверяйте ни одному tool call. Валидируйте:

1. **Type coercion.** Модель может вернуть строку "5" там, где schema требует int. Приводите тип, если это однозначно; иначе отклоняйте.
2. **Enum validation.** Если schema говорит `status in {"open", "closed"}`, а модель выдает `"in_progress"`, отклоняйте с понятной ошибкой.
3. **Required fields.** Отсутствующее обязательное поле -> немедленное error observation обратно модели, а не crash.
4. **Format validation.** Даты, emails, URLs — валидируйте конкретными парсерами, а не regex.

Каждый validation failure должен возвращать structured observation, чтобы модель могла повторить попытку с правильной формой.

### Параллельные вызовы инструментов

Современные providers поддерживают parallel tool calls в одном assistant turn. Цикл:

1. Модель выдает 3 tool calls с разными `tool_use_id`.
2. Runtime выполняет их (параллельно, если они независимы).
3. Каждый результат возвращается как блок `tool_result`, связанный через `tool_use_id`.

Инженерное правило: считайте correlation IDs несущими. Поменяете их местами — получите маршрутизацию результата не к тому tool.

### Sandboxing

Tool execution — это sandbox boundary. См. Lesson 09 подробнее. Коротко: каждый tool должен явно задавать read/write surface, network access, timeout, memory cap. Generic `run_shell(cmd)` — red flag; specific `git_status()` безопаснее.

## Соберите это

`code/main.py` реализует tool registry в production-like форме:

- Validator подмножества JSON Schema (только stdlib).
- Регистрацию tool с description, input schema, timeout и executor.
- Приведение arguments и enum validation.
- Параллельную dispatch tools с correlation IDs.
- Error observations как структурированные строки.

Запустите:

```
python3 code/main.py
```

Трасса показывает mini agent, вызывающий три tools за один turn, с одним намеренно malformed call, который отклоняется с понятной ошибкой, на которую модель может отреагировать.

## Используйте это

У каждого provider своя tool schema — Anthropic, OpenAI, Gemini, Bedrock. Используйте translation layer (OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapter), если нужен multi-provider. BFCL — reference benchmark; запускайте его против своего агента перед shipping, если tool use центральный для продукта.

## Отгрузите это

`outputs/skill-tool-registry.md` генерирует tool catalog, schema и registry для заданного домена задач. Включает description-quality checks (does each tool's description tell the model when to use it?).

## Упражнения

1. Добавьте "no-op" tool, который позволяет модели явно отказаться от использования любого другого tool. Измерьте на BFCL-like hallucination test.
2. Реализуйте argument coercion для int-as-string и float-as-string. Где coercion начинает скрывать реальные bugs?
3. Добавьте per-tool timeout и circuit breaker (refuse the tool for 60s after 3 consecutive failures). Что это меняет в том, как модель восстанавливается?
4. Прочитайте BFCL V4 description. Выберите одну category (например, "multi-turn") и прогоните 10 example prompts через своего agent. Report pass rate.
5. Перенесите stdlib validator на Pydantic или Zod. Что Pydantic/Zod поймали, а toy пропустил?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Function calling | "Tool use" | Вызов tool через structured output с валидированной schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — сохранять tool calls, результаты которых снижают next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | Benchmark 2026 года: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Сигнатура function для модели" | name, description, JSON Schema arguments |
| tool_use_id | "Correlation ID" | Связывает tool call с result; essential для parallel dispatch |
| Hallucination detection | "Know when not to call" | Category V4: отказаться от вызова, когда не подходит ни один tool |
| Argument coercion | "String-to-int repair" | Узкие fixes для predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Граница исполнения tool" | Per-tool read/write surface, network, timeout, memory cap |

## Дополнительное чтение

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — eval benchmark 2026 года
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — production tool schema в Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — function tool type и Guardrails
