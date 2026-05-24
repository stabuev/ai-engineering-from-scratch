# MCP Apps — интерактивные UI-ресурсы через `ui://`

> Текстовый вывод инструмента ограничивает то, что агенты могут показывать. MCP Apps (SEP-1724, официальный статус с 2026-01-26) позволяют инструменту вернуть интерактивный HTML в sandbox, отображаемый inline в Claude Desktop, ChatGPT, Cursor, Goose и VS Code. Дашборды, формы, карты, 3D-сцены — все через одно расширение. В этом уроке рассматриваются схема ресурсов `ui://`, MIME `text/html;profile=mcp-app`, iframe-sandbox протокол postMessage и поверхность безопасности, которая появляется, когда серверу разрешают рендерить HTML.

**Тип:** Build
**Языки:** Python (stdlib, эмиттер UI-ресурсов), HTML (пример приложения)
**Предварительные требования:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Время:** ~75 минут

## Цели обучения

- Возвращать ресурс `ui://` из вызова инструмента и задавать корректные MIME и metadata.
- Объявлять связанный с инструментом UI через `_meta.ui.resourceUri`, `_meta.ui.csp` и `_meta.ui.permissions`.
- Реализовать iframe sandbox postMessage JSON-RPC для связи UI с host.
- Применять CSP и значения permissions-policy по умолчанию, защищающие от атак, исходящих из UI.

## Проблема

Инструмент `visualize_timeline` эпохи 2025 может вернуть "Here are 14 notes organized chronologically: ...". Это абзац. Пользователи на самом деле хотят интерактивную timeline. До MCP Apps варианты были такими: специфичные для клиента widget APIs (Claude artifacts, OpenAI Custom GPT HTML) или вообще без UI.

MCP Apps (SEP-1724, shipped 2026-01-26) стандартизируют контракт. Результат инструмента содержит `resource`, URI которого равен `ui://...`, а MIME равен `text/html;profile=mcp-app`. Host отображает его в iframe с sandbox, ограниченным CSP и без сетевого доступа, если он явно не выдан. UI внутри iframe отправляет сообщения host через небольшой диалект postMessage JSON-RPC.

Каждый совместимый клиент (Claude Desktop, ChatGPT, Goose, VS Code) отображает один и тот же ресурс `ui://` одинаково. Один сервер, один HTML bundle, универсальный UI.

## Концепция

### Схема ресурсов `ui://`

Инструмент возвращает:

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

Затем host вызывает `resources/read` для URI `ui://notes/timeline` и получает:

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox

Host отображает HTML внутри `<iframe>` с sandbox:

- `sandbox="allow-scripts allow-same-origin"` (или строже согласно декларации сервера)
- CSP, объявленным сервером и применяемым через response headers.
- Без cookies, без localStorage из origin host.
- Сетевым доступом, ограниченным `connectSrc` в CSP.

### Протокол postMessage

Iframe общается с host через `window.postMessage`. Небольшой диалект JSON-RPC 2.0:

Всегда фиксируйте `targetOrigin` на точный origin другой стороны, а на принимающей стороне проверяйте `event.origin` по allowlist перед обработкой payload. Никогда не используйте `"*"` ни на одной стороне этого канала — тело переносит вызовы инструментов и чтение ресурсов.

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

Доступные методы на стороне host, которые UI может вызывать:

- `host.callTool(name, arguments)` — вызывает инструмент сервера.
- `host.readResource(uri)` — читает MCP resource.
- `host.getPrompt(name, arguments)` — получает шаблон prompt.
- `host.close()` — закрывает UI.

Каждый вызов все равно проходит через MCP protocol и наследует permissions сервера.

### Permissions

Список `_meta.ui.permissions` запрашивает дополнительные capabilities:

- `camera` — доступ к камере пользователя (используется для UI сканирования документов).
- `microphone` — голосовой ввод.
- `geolocation` — местоположение.
- `network:*` — более широкий сетевой доступ, чем разрешает один `connectSrc`.

Каждое permission — это prompt, который пользователь видит перед рендерингом UI.

### Риски безопасности

HTML в iframe все еще HTML. Новая поверхность атаки:

- **Prompt-injection через UI.** Вредоносный server UI может показать текст, похожий на system message, и обмануть пользователя. Рендеринг host должен заметно отличать server UI от host UI.
- **Exfiltration через `connectSrc`.** Если CSP разрешает `connect-src: *`, UI может отправлять данные куда угодно. Значение по умолчанию должно быть строгим.
- **Clickjacking.** UI перекрывает chrome host. Hosts должны предотвращать манипуляции z-index и принудительно применять правила opacity.
- **Steal focus.** UI забирает keyboard focus и перехватывает следующее сообщение. Hosts должны это перехватывать.

Phase 13 · 15 подробно разбирает это как часть MCP security; этот урок вводит тему.

### Handshake `ui/initialize`

После загрузки iframe отправляет `ui/initialize` через postMessage:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

Host отвечает capabilities и session token. UI использует session token при каждом последующем вызове host.

### Примитивы SDK AppRenderer / AppFrame

SDK ext-apps предоставляет два удобных примитива:

- `AppRenderer` (server side) — оборачивает React / Vue / Solid component и эмитит ресурс `ui://` с правильными MIME и metadata.
- `AppFrame` (client side) — получает ресурс, монтирует iframe и посредничает в postMessage.

Можно использовать их или вручную собрать HTML и JSON-RPC.

### Статус экосистемы

MCP Apps shipped 2026-01-26. Поддержка клиентов на апрель 2026:

- **Claude Desktop.** Полная поддержка с января 2026.
- **ChatGPT.** Полная поддержка через Apps SDK (тот же базовый MCP Apps protocol).
- **Cursor.** Beta; включается в settings.
- **VS Code.** Только Insider builds.
- **Goose.** Полная поддержка.
- **Zed, Windsurf.** В roadmap.

Production-серверы: дашборды, визуализации карт, таблицы данных, конструкторы графиков, IDE preview в sandbox.

## Использование

`code/main.py` расширяет notes server инструментом `visualize_timeline`, который возвращает ресурс `ui://notes/timeline`, плюс handler для `resources/read` на этом URI, который возвращает небольшой, но полный HTML bundle с SVG timeline. HTML шаблонизируется через stdlib — без build system. postMessage набросан в комментариях JS, потому что stdlib не может управлять браузером.

На что смотреть:

- `_meta.ui` в ответе инструмента несет resourceUri, CSP, permissions.
- HTML рендерится без сетевого доступа; все данные inlined.
- JS вызывает `host.callTool` через `window.parent.postMessage` (задокументировано, но неактивно в этой stdlib demo).

## Результат

Этот урок создает `outputs/skill-mcp-apps-spec.md`. Для инструмента, которому полезен интерактивный UI, skill выдает полный контракт MCP Apps: URI `ui://`, CSP, permissions, entrypoints postMessage и security checklist.

## Упражнения

1. Запустите `code/main.py` и изучите сгенерированный HTML. Откройте HTML напрямую в браузере; проверьте, что SVG рендерится. Затем набросайте контракт postMessage, который UI использовал бы для вызова `host.callTool("notes_update", ...)`.

2. Ужесточите CSP: удалите `'unsafe-inline'` и используйте script policy на основе nonce. Что изменится в коде генерации HTML?

3. Добавьте второй UI resource `ui://notes/editor` с формой для редактирования заметки на месте. Когда пользователь отправляет форму, iframe вызывает `host.callTool("notes_update", ...)`.

4. Проведите audit поверхности атаки UI. Где вредоносный сервер мог бы внедрить content? От чего iframe sandbox защищает и от чего нет?

5. Прочитайте спецификацию SEP-1724 и определите одну capability в MCP Apps SDK, которую эта игрушечная реализация не использует. (Hint: синхронизация состояния на уровне component.)

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| MCP Apps | "Интерактивные UI resources" | Расширение SEP-1724, shipped 2026-01-26 |
| `ui://` | "App URI scheme" | Схема ресурсов для UI bundles |
| `text/html;profile=mcp-app` | "The MIME" | Content-type для MCP App HTML |
| Iframe sandbox | "Render container" | Browser sandboxing UI с CSP и permissions |
| postMessage JSON-RPC | "UI-to-host wire" | Небольшой диалект JSON-RPC-over-postMessage для вызовов host |
| `_meta.ui` | "Tool-UI binding" | Metadata, связывающие результат инструмента с UI resource |
| CSP | "Content-Security-Policy" | Объявляет разрешенные источники для scripts, network, styles |
| AppRenderer | "Server SDK primitive" | Преобразует framework component в ресурс `ui://` |
| AppFrame | "Client SDK primitive" | Iframe mount helper, который посредничает в postMessage |
| `ui/initialize` | "Handshake" | Первый postMessage от UI к host |

## Дополнительное чтение

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation и SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — формальный документ спецификации
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — high-level документация
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — launch post за январь 2026
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — справочник SDK в стиле JSDoc
