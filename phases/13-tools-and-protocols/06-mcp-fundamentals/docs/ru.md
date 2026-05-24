# Основы MCP — primitives, lifecycle, база JSON-RPC

> До MCP каждая integration была one-off. Model Context Protocol, впервые выпущенный Anthropic в ноябре 2024 года и теперь сопровождаемый Agentic AI Foundation при Linux Foundation, стандартизует discovery и invocation, чтобы любой client мог говорить с любым server. Spec 2025-11-25 называет шесть primitives (три server, три client), трехфазный lifecycle и wire format JSON-RPC 2.0. Освойте это, и остальная MCP-глава этой phase станет чтением.

**Тип:** теория
**Языки:** Python (stdlib, JSON-RPC parser)
**Предварительные знания:** Phase 13 · 01-05 (интерфейс инструментов и function calling)
**Время:** ~45 минут

## Цели обучения

- Назвать все шесть MCP primitives (tools, resources, prompts на server; roots, sampling, elicitation на client) и дать по одному use case.
- Пройти трехфазный lifecycle (initialize, operation, shutdown) и указать, кто отправляет какое message на каждой phase.
- Парсить и выдавать JSON-RPC 2.0 request, response и notification envelopes.
- Объяснить, что такое capability negotiation при `initialize` и что ломается без него.

## Проблема

До MCP у каждого tool-using agent был собственный protocol. У Cursor была MCP-shaped, но incompatible tool system. Claude Desktop поставлялся с другой. Расширение Copilot в VS Code имело третью. Команда, построившая "Postgres query" tool, писала тот же tool три раза, каждый под API другого host. Reuse требовал копирования code.

Результатом был кембрийский взрыв one-off integrations и потолок скорости ecosystem.

MCP решает это через standardized wire format. Один MCP server работает в каждом MCP client: Claude Desktop, ChatGPT, Cursor, VS Code, Gemini, Goose, Zed, Windsurf, 300+ clients к апрелю 2026 года. 110M загрузок SDK в месяц. 10,000+ public servers. Linux Foundation взяла stewardship в декабре 2025 года под новой Agentic AI Foundation.

Spec revision, используемая в этой phase, — **2025-11-25**. Она добавляет async Tasks (SEP-1686), URL-mode elicitation (SEP-1036), sampling with tools (SEP-1577), incremental scope consent (SEP-835) и OAuth 2.1 resource-indicator semantics. Phase 13 · 09-16 покрывают эти extensions. Этот урок останавливается на base.

## Концепция

### Три server primitives

1. **Tools.** Вызываемые действия. Тот же четырехшаговый цикл из Phase 13 · 01.
2. **Resources.** Открытые данные. Read-only content, адресуемый по URI: `file:///path`, `db://query/...`, custom schemes.
3. **Prompts.** Переиспользуемые templates. Slash-commands в host UI; server предоставляет template, client заполняет arguments.

### Три client primitives

4. **Roots.** Набор URI, к которым server разрешено обращаться. Client объявляет их; server соблюдает границы.
5. **Sampling.** Server просит модель клиента выполнить completion. Это включает server-hosted agent loops без server-side API keys.
6. **Elicitation.** Server запрашивает у пользователя клиента structured input в процессе выполнения. Формы или URL (SEP-1036).

Каждая capability в MCP относится ровно к одной из этих шести. Phase 13 · 10-14 подробно покрывают каждую.

### Wire format: JSON-RPC 2.0

Каждое message — JSON object с этими fields:

- Requests: `{jsonrpc: "2.0", id, method, params}`.
- Responses: `{jsonrpc: "2.0", id, result | error}`.
- Notifications: `{jsonrpc: "2.0", method, params}` — нет `id`, response не ожидается.

Base spec имеет около 15 methods, сгруппированных по primitive. Важные:

- `initialize` / `initialized` (handshake)
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage` (server-to-client)
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### Трехфазный lifecycle

**Phase 1: initialize.**

Client отправляет `initialize` со своими `capabilities` и `clientInfo`. Server отвечает собственными `capabilities`, `serverInfo` и spec version, на которой он говорит. Client отправляет `notifications/initialized`, когда обработал response. С этого момента любая сторона может отправлять requests согласно согласованным capabilities.

**Phase 2: operation.**

Двунаправленно. Client вызывает `tools/list` для discovery, затем `tools/call` для invocation. Server может отправить `sampling/createMessage`, если объявил эту capability. Server может отправить `notifications/tools/list_changed`, когда меняется его tool set. Client может отправить `notifications/roots/list_changed`, когда user меняет root scope.

**Phase 3: shutdown.**

Любая сторона закрывает transport. В MCP нет structured shutdown method; transport (stdio или Streamable HTTP, Phase 13 · 09) несет сигнал конца соединения.

### Capability negotiation

`capabilities` в handshake `initialize` — это contract. Пример от server:

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

Server declares, что он может emit `tools/list_changed` notifications и supports `resources/subscribe`. Client соглашается, объявляя свои:

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

Если client не declares `sampling`, server не должен вызывать `sampling/createMessage`. Симметрично: если server не declares `resources.subscribe`, client не должен пытаться subscribe.

Именно это предотвращает ecosystem drift. Client, который не supports sampling, все еще valid MCP client; server, который не calls `sampling`, все еще valid MCP server. Они просто не используют эту feature вместе.

### Structured content и формы ошибок

`tools/call` возвращает массив `content` из typed blocks: `text`, `image`, `resource`. Phase 13 · 14 добавляет MCP Apps (`ui://` interactive UI) в этот list.

Errors используют JSON-RPC error codes. Дополнения, заданные spec: `-32002` "Resource not found", `-32603` "Internal error", плюс MCP-specific error data в `error.data`.

### Client capabilities vs детали tool call

Частая путаница: `capabilities.tools` означает, поддерживает ли client notifications об изменении списка tools. То, будет ли client вызывать specific tools, — runtime choice, driven by its model, а не capability flag. Capability flag — spec-level contract. Выбор модели ортогонален.

### Почему JSON-RPC, а не REST?

JSON-RPC 2.0 (2010) — легковесный двунаправленный protocol. REST инициируется client. MCP требовал server-initiated messages (sampling, notifications), поэтому JSON-RPC с symmetric request/response shape естественно подошел. JSON-RPC также чисто компонуется поверх stdio и WebSocket/Streamable HTTP без переизобретения request shape HTTP.

## Используйте

`code/main.py` поставляет минимальные JSON-RPC 2.0 parser and emitter, затем вручную проходит sequence `initialize` → `tools/list` → `tools/call` → `shutdown`, печатая каждое message. Real transport нет; только message shapes. Сравните со spec из Further Reading, чтобы проверить каждый envelope.

На что смотреть:

- `initialize` объявляет capabilities в обе стороны; response содержит `serverInfo` и `protocolVersion: "2025-11-25"`.
- `tools/list` возвращает массив `tools`; каждая запись содержит `name`, `description`, `inputSchema`.
- `tools/call` использует `params.name` и `params.arguments`.
- Response `content` — это массив blocks `{type, text}`.

## Отправьте

Этот урок создает `outputs/skill-mcp-handshake-tracer.md`. Получив pcap-style transcript взаимодействия MCP client-server, skill аннотирует каждое message: к какому primitive оно относится, в какой lifecycle phase находится и от какой capability зависит.

## Exercises

1. Запустите `code/main.py`. Найдите строку, где происходит capability negotiation, и опишите, что изменилось бы, если бы server не объявил `tools.listChanged`.

2. Расширьте parser для обработки `notifications/progress`. Message shape: `{method: "notifications/progress", params: {progressToken, progress, total}}`. Emit it while a long-running `tools/call` is in progress и подтвердите, что client handler показал бы progress bar.

3. Прочитайте spec MCP 2025-11-25 от начала до конца — весь document около 80 pages. Найдите один capability flag, который большинству servers НЕ нужен. Подсказка: он связан с resource subscription.

4. Нарисуйте на бумаге primitive, к которому принадлежала бы hypothetical feature "cron job". (Подсказка: server хочет, чтобы client invoked it at a scheduled time. Сегодня ни одна из шести primitives не подходит.) В roadmap MCP 2026 года есть draft SEP для этого.

5. Распарсите один session log из open MCP server на GitHub. Посчитайте messages типа request, response и notification. Вычислите, какая доля traffic относится к lifecycle vs operation.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | Open protocol для discovery и invocation между model и tool |
| Server primitive | "Что server exposes" | tools (actions), resources (data), prompts (templates) |
| Client primitive | "Что client lets servers use" | roots (scope), sampling (LLM callbacks), elicitation (user input) |
| JSON-RPC 2.0 | "Wire format" | Симметричные envelopes request/response/notification |
| `initialize` handshake | "Capability negotiation" | Первая пара messages; servers и clients объявляют поддерживаемые features |
| `tools/list` | "Discovery" | Client запрашивает у server текущий tool set |
| `tools/call` | "Invocation" | Client просит server выполнить tool с arguments |
| `notifications/*_changed` | "Mutation events" | Server сообщает client, что список primitive изменился |
| Content block | "Typed result" | `{type: "text" | "image" | "resource" | "ui_resource"}` in tool result |
| SEP | "Spec Evolution Proposal" | Именованное draft proposal (например, SEP-1686 для async Tasks) |

## Дополнительное чтение

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — канонический документ spec
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — mental model шести primitives
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — launch post ноября 2024 года
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — годовая ретроспектива и изменения spec 2025-11-25
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — summary SEP-1686, 1036, 1577, 835 и 1724
