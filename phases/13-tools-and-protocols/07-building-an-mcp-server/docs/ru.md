# Создание MCP-сервера — Python + TypeScript SDK

> Большинство руководств по MCP показывают только hello-world через stdio. Настоящий сервер предоставляет tools, resources и prompts, обрабатывает согласование capabilities, выдает структурированные ошибки и ведет себя одинаково в разных SDK. В этом уроке мы строим notes server от начала до конца: stdlib stdio transport, JSON-RPC dispatch, три серверных примитива и стиль чистых функций, который затем без переписывания логики переносится либо в FastMCP из Python SDK, либо в TypeScript SDK.

**Тип:** Build
**Языки:** Python (stdlib, stdio MCP server)
**Предварительные требования:** Phase 13 · 06 (MCP fundamentals)
**Время:** ~75 минут

## Цели обучения

- Реализовать методы `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list` и `prompts/get`.
- Написать dispatch loop, который читает JSON-RPC сообщения из stdin и пишет ответы в stdout.
- Выдавать структурированные ответы с ошибками по спецификации JSON-RPC 2.0 и дополнительным кодам MCP.
- Перенести stdlib-реализацию в FastMCP (Python SDK) или TypeScript SDK без переписывания логики tools.

## Проблема

Прежде чем использовать remote transport (Phase 13 · 09) или auth layer (Phase 13 · 16), нужен аккуратный локальный сервер. Локальный означает stdio: клиент запускает сервер как дочерний процесс, а сообщения идут через stdin/stdout с разделением по строкам.

Спецификация 2025-11-25 предписывает кодировать stdio-сообщения как JSON-объекты с явным разделителем `\n`. Здесь нет SSE; SSE был старым remote mode и удаляется в середине 2026 года (MCP-сервер Atlassian Rovo удаляет его 30 июня 2026 года; Keboola — 1 апреля 2026 года). Для stdio весь wire format — один JSON-объект на строку.

Notes server — удачная форма, потому что он задействует все три серверных примитива. Tools выполняют мутации (`notes_create`). Resources предоставляют данные (`notes://{id}`). Prompts поставляют шаблоны (`review_note`). Форма этого урока обобщается на любой домен.

## Концепция

### Dispatch loop

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

Три правила:

- Не печатайте в stdout ничего, что не является JSON-RPC envelope. Debug logs идут в stderr.
- На каждый request ДОЛЖЕН быть отправлен response с тем же `id`.
- На notifications НЕЛЬЗЯ отвечать.

### Реализация `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

Объявляйте только то, что поддерживаете. Клиент опирается на набор capabilities, чтобы включать или скрывать функции.

### Реализация `tools/list` и `tools/call`

`tools/list` возвращает `{tools: [...]}`, где каждая запись имеет `name`, `description`, `inputSchema`. `tools/call` принимает `{name, arguments}` и возвращает `{content: [blocks], isError: bool}`.

Content blocks имеют типы. Самые распространенные:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Ошибки tools бывают двух видов. Protocol-level errors (unknown method, bad params) — это JSON-RPC errors. Tool-level errors (valid call, но tool завершился неуспешно) возвращаются как `{content: [...], isError: true}`. Так модель видит сбой в своем контексте.

### Реализация resources

Resources по замыслу read-only. `resources/list` возвращает manifest; `resources/read` возвращает content. URI могут быть `file://...`, `http://...` или custom scheme вроде `notes://`.

Когда вы предоставляете данные как resource, а не как tool:

- Модель не "вызывает" его; клиент может добавить его в контекст по запросу пользователя.
- Subscriptions позволяют серверу отправлять updates, когда resource меняется (Phase 13 · 10).
- Phase 13 · 14 расширяет это через `ui://` для interactive resources.

### Реализация prompts

Prompts — это шаблоны с именованными arguments. Host показывает их как slash-commands. Prompt `review_note` может принимать argument `note_id` и создавать multi-message prompt template, который клиент передает своей модели.

### Тонкости stdio transport

- Newline-delimited JSON. Никакого length-prefixed framing.
- Не буферизуйте. Вызывайте `sys.stdout.flush()` после каждой записи.
- Клиент управляет временем жизни. Когда stdin закрывается (EOF), завершайтесь корректно.
- Не обрабатывайте SIGPIPE молча; логируйте и выходите.

### Annotations

Каждый tool может иметь `annotations`, описывающие свойства безопасности:

- `readOnlyHint: true` — чистое чтение, безопасно повторять.
- `destructiveHint: true` — необратимые side effects; клиент должен запросить подтверждение.
- `idempotentHint: true` — одинаковые inputs дают одинаковые outputs.
- `openWorldHint: true` — взаимодействует с внешними системами.

Клиент использует это для UX (confirmation dialogs, status indicators) и routing (Phase 13 · 17).

### Путь перехода

Stdlib server в `code/main.py` занимает около 180 строк. FastMCP (Python) сворачивает ту же логику в decorator-style:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK имеет аналогичную форму. Переход можно сделать drop-in, когда будете готовы; концепции (capabilities, dispatch, content blocks) остаются теми же.

## Использование

`code/main.py` — полный notes MCP server поверх stdio, только stdlib. Он обрабатывает `initialize`, `tools/list`, `tools/call` для трех tools (`notes_list`, `notes_search`, `notes_create`), `resources/list` и `resources/read` для каждой note, а также prompt `review_note`. Его можно запускать, передавая JSON-RPC messages через pipe:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

На что смотреть:

- Dispatcher — это `dict[str, Callable]`, индексированный по имени method.
- Каждый tool executor возвращает список content blocks, а не голую строку.
- `isError: true` выставляется, когда executor выбрасывает исключение.

## Ship It

Этот урок создает `outputs/skill-mcp-server-scaffolder.md`. По заданному domain (notes, tickets, files, database) skill scaffolds MCP server с правильным разделением tools / resources / prompts и путем перехода на SDK.

## Упражнения

1. Запустите `code/main.py` и управляйте им вручную собранными JSON-RPC messages. Выполните `notes_create`, затем `resources/read`, чтобы получить новую note.

2. Добавьте tool `notes_delete` с `annotations: {destructiveHint: true}`. Проверьте, что клиент показал бы confirmation dialog (для этого нужен настоящий host; Claude Desktop подходит).

3. Реализуйте `resources/subscribe`, чтобы сервер отправлял `notifications/resources/updated` при каждом изменении note. Добавьте keepalive task.

4. Перенесите сервер на FastMCP. Python-файл должен сократиться до менее чем 80 строк. Wire behavior должен быть идентичным; проверьте тем же JSON-RPC test harness.

5. Прочитайте раздел `server/tools` в спецификации и найдите одно поле tool definition, не реализованное в сервере этого урока. (Подсказка: их несколько; выберите одно и добавьте его.)

## Ключевые термины

| Термин | Как говорят | Что это значит на самом деле |
|------|----------------|------------------------|
| MCP server | "То, что предоставляет tools" | Процесс, который говорит на MCP JSON-RPC через stdio или HTTP |
| stdio transport | "Модель дочернего процесса" | Сервер запускается клиентом; общается через stdin/stdout |
| Dispatcher | "Router методов" | Отображение имени JSON-RPC method на handler function |
| Content block | "Фрагмент результата tool" | Типизированный элемент в массиве `content` ответа tool |
| `isError` | "Failure на уровне tool" | Сигнализирует, что tool завершился ошибкой; отличается от JSON-RPC error |
| Annotations | "Подсказки безопасности" | Флаги readOnly / destructive / idempotent / openWorld |
| FastMCP | "Python SDK" | Высокоуровневый decorator-based framework поверх MCP protocol |
| Resource URI | "Адресуемые данные" | `file://`, `db://` или custom scheme, идентифицирующий resource |
| Prompt template | "Краткий slash-command" | Server-supplied template с slots для arguments в host UIs |
| Capability declaration | "Feature toggle" | Per-primitive flags, объявленные в `initialize` |

## Дополнительное чтение

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — reference implementation на Python
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — параллельная TS-реализация
- [FastMCP — server framework](https://gofastmcp.com/) — decorator-style Python API для MCP servers
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — end-to-end tutorial с использованием любого из SDK
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — полный справочник по messages `tools/*`
