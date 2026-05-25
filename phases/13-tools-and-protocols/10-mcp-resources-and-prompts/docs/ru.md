# MCP Resources и Prompts — контекст за пределами tools

> Tools получают 90 процентов внимания в MCP. Два других серверных примитива решают другие задачи. Resources предоставляют данные для чтения; prompts предоставляют переиспользуемые шаблоны как slash-commands. Многие servers должны использовать resources вместо оборачивания чтения в tools, и prompts вместо жестко зашитых workflows в client prompts. Этот урок формулирует правило выбора и проходит по сообщениям `resources/*` и `prompts/*`.

**Тип:** Build
**Языки:** Python (stdlib, resource + prompt handler)
**Предварительные требования:** Phase 13 · 07 (MCP server)
**Время:** ~45 минут

## Цели обучения

- Решать, предоставлять ли capability как tool, resource или prompt для заданного domain.
- Реализовать `resources/list`, `resources/read`, `resources/subscribe` и обработать `notifications/resources/updated`.
- Реализовать `prompts/list` и `prompts/get` с templates аргументов.
- Понимать, когда host показывает prompts как slash-commands, а когда как автоматически добавленный context.

## Проблема

Наивный MCP server для notes app предоставляет все как tools: `notes_read`, `notes_list`, `notes_search`. Это оборачивает каждый доступ к данным в tool call под управлением модели. Последствия:

- Модель должна решать, вызывать ли `notes_read` для каждого query, которому может помочь context.
- Read-only content нельзя подписать или отправить stream в side panel host.
- Client UIs (панель прикрепления resources в Claude Desktop, picker "Include file" в Cursor) не могут показать эти данные.

Правильное разделение: предоставляйте данные как resource, mutating или computed actions как tools, переиспользуемые multi-step workflows как prompts. У каждого primitive есть свой UX affordance и паттерн доступа.

## Концепция

### Tools, resources и prompts — правило выбора

| Capability | Primitive |
|------------|-----------|
| Пользователь хочет искать, фильтровать или преобразовывать данные | tool |
| Пользователь хочет, чтобы host включал эти данные как context | resource |
| Пользователь хочет шаблонный workflow, который можно повторять | prompt |

Ориентир: если модели полезно вызывать это для каждого связанного query, это tool. Если пользователю полезно прикрепить это к conversation, это resource. Если единицей повторного использования является целый multi-step workflow, это prompt.

### Resources

`resources/list` возвращает `{resources: [{uri, name, mimeType, description?}]}`. `resources/read` принимает `{uri}` и возвращает `{contents: [{uri, mimeType, text | blob}]}`.

URI могут быть чем угодно адресуемым:

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (custom scheme)
- `memory://session-2026-04-22/recent` (server-specific)

`contents[]` поддерживает и text, и binary. Binary использует `blob` как base64-encoded string плюс `mimeType`.

### Resource subscriptions

Объявите `{resources: {subscribe: true}}` в capabilities. Client вызывает `resources/subscribe {uri}`. Server отправляет `notifications/resources/updated {uri}`, когда resource меняется. Client перечитывает.

Use case: notes server, у которого resources — это files на диске; file watcher запускает update notifications; Claude Desktop повторно подтягивает file в context, когда его редактируют вне host.

### Resource templates (добавление 2025-11-25)

`resourceTemplates` позволяют предоставить parameterized URI pattern: `notes://{id}`, где `id` — цель completion. Client может выполнять autocomplete ids в resource picker.

### Prompts

`prompts/list` возвращает `{prompts: [{name, description, arguments?}]}`. `prompts/get` принимает `{name, arguments}` и возвращает `{description, messages: [{role, content}]}`.

Prompt — это template, который заполняется в список messages, передаваемый host своей модели. Например, prompt `code_review` принимает argument `file_path` и возвращает sequence из трех messages: system message, user message с телом file и стартовое assistant message с reasoning template.

### Hosts и prompts

Claude Desktop, VS Code и Cursor показывают prompts как slash-commands в chat UI. Пользователь вводит `/code_review` и выбирает arguments из form. Prompt server — это contract между "user shortcut" и "полным prompt, отправленным модели".

Не каждый client пока поддерживает prompts — проверяйте capability negotiation. Server с объявленной prompt capability, но client без поддержки prompts, просто не увидит slash commands.

### Notification "list changed"

И resources, и prompts отправляют `notifications/list_changed`, когда их набор меняется. Notes server, который только что импортировал 20 новых notes, отправляет `notifications/resources/list_changed`; client повторно вызывает `resources/list`, чтобы подобрать additions.

### Соглашения по content type

Для text: `mimeType: "text/plain"`, `text/markdown`, `application/json`.
Для binary: `image/png`, `application/pdf` плюс поле `blob`.
Для MCP Apps (Lesson 14): `text/html;profile=mcp-app` в `ui://` URI.

### Dynamic resources

Resource URI не обязан соответствовать static file. `notes://recent` может возвращать пять последних notes при каждом read. `db://query/users/active` может выполнять parameterized query. Server свободен вычислять content dynamically.

Правило: если client может cache by URI, URI должен быть stable. Если computation one-shot, URI должен включать timestamp или nonce, чтобы client cache не устаревал незаметно.

### Subscriptions и polling

Clients с поддержкой subscriptions получают server push через `notifications/resources/updated`. Pre-subscription clients или hosts, которые это не поддерживают, poll через повторное чтение. Оба варианта spec-compliant. Capability declaration server сообщает client, что поддерживается.

Стоимость subscriptions: per-session state на server (кто на что подписан). Держите subscribed set bounded; disconnected clients должны истекать по timeout.

### Prompts и system prompts

Prompts в MCP — не system prompts. System prompt host (его собственные operating instructions) и MCP prompts (templates от server, invoked by user) живут side by side. Well-behaved client никогда не позволяет server prompt override its own system prompt; он накладывает их слоями.

## Использование

`code/main.py` расширяет notes server из Lesson 07:

- Per-note resources (`notes://note-1`, etc.) с поддержкой `resources/subscribe`.
- Prompt `review_note`, который renders в template из трех messages.
- Симуляция file watcher, которая emits `notifications/resources/updated`, когда note modified.
- Dynamic resource `notes://recent`, который всегда возвращает пять последних notes.

Запустите demo, чтобы увидеть полный flow.

## Результат

Этот урок создает `outputs/skill-primitive-splitter.md`. По proposed MCP server skill классифицирует каждую capability как tool / resource / prompt с обоснованием.

## Упражнения

1. Запустите `code/main.py`. Посмотрите initial resource list, затем запустите редактирование note и проверьте, что событие `notifications/resources/updated` срабатывает.

2. Добавьте emitter `resources/list_changed`: когда создается new note, отправьте notification, чтобы clients повторно выполнили discovery.

3. Спроектируйте три prompts для GitHub MCP server: `summarize_pr`, `triage_issue`, `release_notes`. Каждый с argument schemas. Prompt body должен запускаться без дальнейших правок.

4. Возьмите существующий tool в server из Lesson 07 и классифицируйте, должен ли он остаться tool или быть разделен на пару resource plus tool. Обоснуйте одним предложением.

5. Прочитайте разделы `server/resources` и `server/prompts` в спецификации. Найдите одно поле в `resources/read`, которое редко заполняется, но поддерживается spec. Подсказка: посмотрите на `_meta` в resource content.

## Ключевые термины

| Термин | Как говорят | Что это значит на самом деле |
|------|----------------|------------------------|
| Resource | "Открытые данные" | URI-addressable content, который host может читать |
| Resource URI | "Указатель на данные" | Identifier со scheme-prefix (`file://`, `notes://`, etc.) |
| `resources/subscribe` | "Следить за изменениями" | Server-push updates для specific URI, на которые client подписался |
| `notifications/resources/updated` | "Resource changed" | Signal клиенту, что subscribed resource получил новый content |
| Resource template | "Parameterized URI" | URI pattern с completion hints для picker в host |
| Prompt | "Slash-command template" | Named multi-message template с slots для arguments |
| Prompt arguments | "Template inputs" | Typed parameters, которые host собирает перед rendering |
| `prompts/get` | "Render template" | Server возвращает заполненный список messages |
| Content block | "Typed chunk" | `{type: text | image | resource | ui_resource}` |
| Slash-command UX | "User shortcut" | Host показывает prompts как команды, начинающиеся с `/` |

## Дополнительное чтение

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URIs, subscriptions и templates
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt templates и интеграция со slash-commands
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — полный reference сообщений `resources/*`
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — полный reference сообщений `prompts/*`
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — community guide, расширяющий официальные docs
