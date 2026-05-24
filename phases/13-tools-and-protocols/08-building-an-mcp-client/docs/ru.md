# Создание MCP-клиента — обнаружение, вызовы, управление сессиями

> Большая часть материалов по MCP показывает руководства по серверам и почти не объясняет клиент. Именно в коде клиента живет сложная оркестрация: запуск процессов, согласование capabilities, объединение списков tools от нескольких серверов, callbacks для sampling, переподключение и разрешение конфликтов имен. В этом уроке мы строим multi-server client, который поднимает три разных MCP-сервера в одно плоское пространство имен tools для модели.

**Тип:** Build
**Языки:** Python (stdlib, multi-server MCP client)
**Предварительные требования:** Phase 13 · 07 (создание MCP-сервера)
**Время:** ~75 минут

## Цели обучения

- Запустить MCP-сервер как дочерний процесс, завершить `initialize` и отправить `notifications/initialized`.
- Поддерживать состояние сессии для каждого сервера (capabilities, список tools, ids последних увиденных notifications).
- Объединять списки tools от нескольких серверов в одно пространство имен с обработкой конфликтов.
- Направлять tool call на сервер, которому он принадлежит, и собирать ответ обратно.

## Проблема

Настоящий agent host (Claude Desktop, Cursor, Goose, Gemini CLI) загружает сразу несколько MCP-серверов. У пользователя могут одновременно работать filesystem server, Postgres server и GitHub server. Задача клиента:

1. Запустить каждый сервер.
2. Выполнить handshake с каждым независимо.
3. Вызвать `tools/list` на каждом и свести результат в плоский список.
4. Когда модель выдает `notes_search`, найти его в объединенном пространстве имен и направить на правильный сервер.
5. Обрабатывать notifications от любого сервера (`tools/list_changed`) без блокировки.
6. Переподключаться при сбое transport.

Ручная реализация всего этого и отделяет игрушечный пример от пригодного в работе клиента. Официальные SDK это оборачивают, но mental model должен быть вашим.

## Концепция

### Запуск дочерних процессов

`subprocess.Popen` с `stdin=PIPE, stdout=PIPE, stderr=PIPE`. Установите `bufsize=1` и используйте текстовый режим для построчного чтения. Каждый сервер — отдельный процесс; клиент держит по одному handle `Popen` на сервер.

### Состояние сессии на каждый сервер

Объект `Session` для каждого сервера хранит:

- `process` — handle `Popen`.
- `capabilities` — что сервер объявил в `initialize`.
- `tools` — последний результат `tools/list`.
- `pending` — отображение request id на promise/future, ожидающий response.

Requests по природе async; `tools/call`, отправленный серверу A, не должен блокироваться, пока сервер B находится в середине вызова. Используйте либо потоки с очередями, либо asyncio.

### Объединенное пространство имен

Когда клиент видит общий список tools, имена могут конфликтовать. Два сервера могут оба предоставлять `search`. У клиента есть три варианта:

1. **Префикс по имени сервера.** `notes/search`, `files/search`. Понятно, но некрасиво.
2. **Молчаливый first-come.** Более поздний `search` переопределяет ранний. Рискованно; скрывает конфликты.
3. **Отклонение конфликта.** Отказаться загружать второй сервер; уведомить пользователя. Самый безопасный вариант для security-sensitive hosts.

Claude Desktop использует префикс по серверу. Cursor использует отклонение конфликта с понятной ошибкой. VS Code MCP тоже принимает префикс по серверу.

### Routing

После объединения dispatch table отображает `tool_name -> session`. Модель выдает вызов по имени; клиент находит сессию и пишет message `tools/call` в stdin этого сервера, затем ожидает response.

### Callback для sampling

Если сервер объявил capability `sampling` в `initialize`, он может отправить `sampling/createMessage`, прося клиента запустить свой LLM. Клиент должен:

1. Блокировать дальнейшие requests к этому серверу, пока sample не завершится, или использовать pipeline, если implementation поддерживает concurrency.
2. Вызвать своего LLM provider.
3. Отправить response обратно серверу.

Lesson 11 разбирает sampling end-to-end. Этот урок оставляет stub для полноты.

### Обработка notifications

`notifications/tools/list_changed` означает: повторно вызвать `tools/list`. `notifications/resources/updated` означает: перечитать resource, если он используется. Notifications не должны порождать responses — не пытайтесь их подтверждать.

Распространенная ошибка клиента: блокировать цикл чтения на `tools/call`, пока notification лежит в stream. Используйте фоновый поток чтения, который кладет каждое message в очередь; основной поток извлекает их и направляет обработчику.

### Reconnection

Transport может сломаться: сервер упал, ОС завершила процесс, stdio pipe оборвался. Клиент обнаруживает EOF на stdout и считает session dead. Варианты:

- Молча перезапустить сервер и заново выполнить handshake. Подходит для чисто read-only servers.
- Показать failure пользователю. Подходит для stateful servers с user-visible sessions.

Phase 13 · 09 рассматривает семантику переподключения Streamable HTTP; stdio проще.

### Keepalive и session id

Streamable HTTP использует header `Mcp-Session-Id`. В stdio нет session id — идентичность процесса и есть session. Keepalive pings необязательны; stdio pipes не ломаются от бездействия.

## Использование

`code/main.py` запускает три simulated MCP servers как subprocesses, выполняет handshake с каждым, объединяет их списки tools и направляет tool calls к нужному серверу. Эти "servers" на самом деле другие процессы Python с игрушечными responders (без настоящего LLM). Запустите, чтобы увидеть:

- Три initializations, у каждого свой набор capabilities.
- Три результата `tools/list`, объединенные в пространство имен из 7 tools.
- Решение о routing на основе имени tool.
- Конфликт, предотвращенный префиксом пространства имен.

На что смотреть:

- Dataclass `Session` аккуратно хранит состояние на каждый сервер.
- Фоновый поток чтения извлекает каждую строку из stdout без блокировки основного потока.
- Dispatch table — простой `dict[str, Session]`.
- Обработка конфликтов явная: когда два сервера объявляют одно имя, более поздний переименовывается с префиксом.

## Ship It

Этот урок создает `outputs/skill-mcp-client-harness.md`. По декларативному списку MCP-серверов (name, command, args) skill создает harness, который запускает их, объединяет списки tools и поставляет routing function с разрешением конфликтов.

## Упражнения

1. Запустите `code/main.py` и посмотрите log запуска серверов. Убейте один из simulated server processes через SIGTERM и наблюдайте, как клиент обнаруживает EOF и помечает эту session как dead.

2. Реализуйте префиксы пространства имен. Когда два сервера предоставляют `search`, переименуйте второй в `<server>/search`. Обновите dispatch table и проверьте, что tool calls направляются правильно.

3. Добавьте backoff в стиле connection pool для перезапуска сервера: exponential backoff при последовательных failures, ограничение в 30 seconds, notification пользователю после трех failures.

4. Набросайте клиент, который поддерживает 100 concurrent MCP servers. Какая data structure заменит простой dispatch dict? (Подсказка: trie для префиксов пространства имен плюс метрика tool-count-per-server.)

5. Перенесите client на официальный MCP Python SDK. SDK оборачивает `stdio_client` и `ClientSession`. Код должен сократиться с ~200 строк до ~40 строк, сохранив multi-server routing.

## Ключевые термины

| Термин | Как говорят | Что это значит на самом деле |
|------|----------------|------------------------|
| MCP client | "Agent host" | Процесс, который запускает серверы и оркестрирует tool calls |
| Session | "Состояние на сервер" | Capabilities, список tools и учет pending requests |
| Merged namespace | "Один список tools" | Плоский набор имен tools по всем активным серверам |
| Namespace collision | "Два сервера, один tool" | Клиент должен добавить префикс, отклонить конфликт или принять первый duplicate |
| Routing | "Кому идет этот вызов?" | Dispatch от имени tool к серверу-владельцу |
| Background reader | "Неблокирующий stdout" | Thread или task, который вычитывает stdout сервера в очередь |
| Sampling callback | "LLM-as-a-service" | Handler клиента для `sampling/createMessage` от сервера |
| `notifications/*_changed` | "Примитив изменился" | Signal, что клиент должен заново выполнить discovery или перечитать данные |
| Reconnection policy | "Когда сервер умер" | Семантика restart при transport failure |
| Stdio session | "Процесс = session" | Нет session id; lifetime дочернего процесса и есть session |

## Дополнительное чтение

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — каноническое поведение клиента
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — hello-world tutorial клиента с Python SDK
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — reference по `ClientSession` и `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — параллельная реализация на TS
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — как VS Code multiplexes несколько MCP-серверов в одном editor host
