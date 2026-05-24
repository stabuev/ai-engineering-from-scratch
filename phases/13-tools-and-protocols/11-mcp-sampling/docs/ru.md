# MCP Sampling — LLM completions и agent loops по запросу сервера

> Большинство MCP servers — простые executors: принять arguments, выполнить code, вернуть content. Sampling позволяет server развернуть направление: он просит LLM клиента принять решение. Это включает agent loops, размещенные на server, без model credentials у server. SEP-1577, merged in 2025-11-25, добавил tools внутри sampling requests, чтобы loop мог включать более глубокое reasoning. Drift-risk note: форма SEP-1577 tool-in-sampling была экспериментальной до Q1 2026 и все еще стабилизируется в SDK APIs.

**Тип:** Build
**Языки:** Python (stdlib, sampling harness)
**Предварительные требования:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources и prompts)
**Время:** ~75 минут

## Цели обучения

- Объяснить, что решает `sampling/createMessage` (server-hosted loops без server-side API keys).
- Реализовать server, который просит client сделать sample по multi-turn prompt и возвращает completion.
- Использовать `modelPreferences` (приоритеты cost / speed / intelligence), чтобы направлять выбор модели клиентом.
- Построить tool `summarize_repo`, который выполняет внутренние итерации через sampling вместо hard-coding behavior.

## Проблема

Полезный MCP server для code-summarization workflow должен: пройти file tree, выбрать, какие files читать, синтезировать summary и вернуть результат. Где происходит LLM reasoning?

Вариант A: server вызывает свой собственный LLM. Нужен API key, billing на стороне server, дорого per user.

Вариант B: server возвращает raw content; agent клиента выполняет reasoning. Работает, но переносит server logic в client prompt, что хрупко.

Вариант C: server спрашивает LLM клиента через `sampling/createMessage`. Server сохраняет algorithm (какие files читать, сколько passes делать), а client сохраняет billing и model choice. У server вообще нет credentials.

Sampling — это вариант C. Это механизм, с помощью которого trusted server может разместить agent loop, не становясь полноценным LLM host.

## Концепция

### Request `sampling/createMessage`

Server отправляет:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Client запускает свой LLM и возвращает:

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

Три floats, сумма которых 1.0:

- `costPriority`: предпочитать более дешевые models.
- `speedPriority`: предпочитать более быстрые models.
- `intelligencePriority`: предпочитать более мощные models.

Плюс `hints`: named models, которые server предпочитает. Client может учитывать hints или игнорировать их; user config клиента всегда главнее.

### `includeContext`

Три значения:

- `"none"` — только server-supplied messages. Default.
- `"thisServer"` — включить prior messages из session этого server.
- `"allServers"` — включить весь session context.

`includeContext` soft-deprecated as of 2025-11-25, потому что он leaks cross-server context, что является security concern. Предпочитайте `"none"` и передавайте explicit context в messages.

### Sampling с tools (SEP-1577)

Новое в 2025-11-25: sampling request может включать массив `tools`. Client запускает полный tool-calling loop с этими tools. Это позволяет server размещать ReAct-style agent loop через модель клиента.

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

Client loops: делает sample, выполняет tool при вызове, снова делает sample, возвращает final assistant message. Это было экспериментальным до Q1 2026; SDK signatures may still drift. При реализации сверяйтесь с разделом client/sampling спецификации 2025-11-25.

### Human-in-the-loop

Client ДОЛЖЕН показать пользователю, что server просит model сделать, перед запуском sample. Malicious server может использовать sampling, чтобы манипулировать user's session ("сказать пользователю X, чтобы он нажал Y"). Claude Desktop, VS Code и Cursor показывают sampling requests как confirmation dialog, который пользователь может отклонить.

Консенсус 2026 года: sampling без human confirmation — red flag. Gateways (Phase 13 · 17) могут auto-approve low-risk sampling и автоматически отклонять все подозрительное.

### Server-hosted loops без API keys

Канонический use case: code-summarization MCP server без собственного LLM access. Он делает:

1. Обходит repo structure.
2. Вызывает `sampling/createMessage` с "Выбери пять файлов, которые вероятнее всего описывают назначение этого repo."
3. Читает эти files.
4. Вызывает `sampling/createMessage` с contents этих files и "Суммаризируй repo в 3 paragraphs."
5. Возвращает summary как результат `tools/call`.

Server никогда не касается LLM API. Пользователь client платит за completions своими credentials.

### Safety risks (Unit 42 disclosure, 2026 Q1)

- **Covert sampling.** Tool, который всегда вызывает sampling с "ответь email пользователя из session context." Phase 13 · 15 разбирает attack vectors.
- **Resource theft via sampling.** Server просит client summarizе payload атакующего и списывает стоимость на пользователя.
- **Loop bombs.** Server вызывает sampling в tight loop. Clients ДОЛЖНЫ enforcing per-session rate limits.

## Использование

`code/main.py` поставляет fake server-to-client sampling harness. Simulated tool "summarize_repo" выполняет два раунда sampling (pick-files, затем summarize), а fake client возвращает canned responses. Harness показывает:

- Server отправляет `sampling/createMessage` с `modelPreferences`.
- Client возвращает completion.
- Server продолжает свой loop.
- Rate limiter ограничивает общее число sampling calls на один tool invocation.

На что смотреть:

- Server предоставляет только один tool (`summarize_repo`); все reasoning происходит в sampling calls.
- Model preferences взвешивают выбор модели клиента; hints перечисляют preferred models.
- Loop завершается на `stopReason: "endTurn"`.
- Лимит `max_samples_per_tool = 5` ловит runaway loop.

## Ship It

Этот урок создает `outputs/skill-sampling-loop-designer.md`. По server-side algorithm, которому нужны LLM calls (research, summarization, planning), skill проектирует sampling-based implementation с правильными `modelPreferences`, rate limits и safety confirmations.

## Упражнения

1. Запустите `code/main.py`. Измените `max_samples_per_tool` на 2 и наблюдайте rate-limit cut-off.

2. Реализуйте вариант SEP-1577 tool-in-sampling: sampling request несет массив `tools`. Проверьте, что client-side loop выполняет эти tools перед возвратом final completion. Drift risk: SDK signatures могут еще меняться до H1 2026.

3. Добавьте human-in-the-loop confirmation: перед первым `sampling/createMessage` от server остановитесь и дождитесь user approval. Отклоненные calls возвращают typed refusal.

4. Добавьте per-user rate limiter, indexed by client session. Same-server loops от одного пользователя должны делить budget.

5. Спроектируйте tool `summarize_pdf`, который использует sampling, чтобы выбрать chunks для включения. Набросайте отправляемые messages. Как `modelPreferences.intelligencePriority` меняет behavior при 0.1 vs 0.9?

## Ключевые термины

| Термин | Как говорят | Что это значит на самом деле |
|------|----------------|------------------------|
| Sampling | "LLM call от server к client" | Server просит модель client дать completion |
| `sampling/createMessage` | "Метод" | JSON-RPC method для sampling requests |
| `modelPreferences` | "Приоритеты модели" | Веса cost / speed / intelligence плюс name hints |
| `includeContext` | "Cross-session leakage" | Soft-deprecated mode включения context |
| SEP-1577 | "Tools в sampling" | Разрешает tools внутри sampling для server-hosted ReAct |
| Human-in-the-loop | "Пользователь подтверждает" | Client показывает sampling request пользователю перед запуском |
| Loop bomb | "Runaway sampling" | Бесконечный sampling loop на стороне server; client должен rate-limit |
| Covert sampling | "Скрытое reasoning" | Malicious server скрывает intent в sampling prompts |
| Resource theft | "Использование LLM budget пользователя" | Server заставляет client тратить budget на ненужный sampling |
| `stopReason` | "Почему generation остановилась" | `endTurn`, `stopSequence` или `maxTokens` |

## Дополнительное чтение

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — high-level overview sampling
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — каноническая форма `sampling/createMessage`
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — Spec Evolution Proposal для tools in sampling (experimental)
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — паттерны covert sampling и resource theft
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — walk-through с client-side code samples
