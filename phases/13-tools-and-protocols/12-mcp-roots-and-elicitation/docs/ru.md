# Roots и Elicitation — scope и пользовательский ввод в середине вызова

> Hard-coded paths ломаются, как только пользователь открывает другой project. Pre-filled tool arguments ломаются, когда пользователь недоуточняет запрос. Roots ограничивают server набором URI под контролем пользователя; elicitation ставит mid-tool-call на паузу, чтобы спросить у пользователя structured input через form или URL. Два client primitives, два исправления для распространенных MCP failure modes. SEP-1036 (URL-mode elicitation, 2025-11-25) был экспериментальным до H1 2026 — проверяйте версии SDK, прежде чем на него полагаться.

**Тип:** Build
**Языки:** Python (stdlib, roots + elicitation demo)
**Предварительные требования:** Phase 13 · 07 (MCP server)
**Время:** ~45 минут

## Цели обучения

- Объявлять `roots` и отвечать на `notifications/roots/list_changed`.
- Ограничивать file operations server URI внутри declared root set.
- Использовать `elicitation/create`, чтобы запросить у пользователя confirmation или structured input mid-tool-call.
- Выбирать между form-mode и URL-mode elicitation (последний experimental; drift-risk noted).

## Проблема

Две конкретные failure, с которыми notes MCP server сталкивается в production.

**Сломанное предположение о path.** Server написан под `~/notes`. У пользователя на другой машине notes лежат в `~/Documents/Notes`; tool call либо тихо fails (file не найден), либо, что хуже, пишет не туда.

**Missing argument, который знает пользователь.** Пользователь просит "удали старую note с TPS report". Модель вызывает `notes_delete(title: "TPS report")`, но есть три matching notes: из 2023, 2024 и 2025 годов. Tool не может угадывать. Failure с "ambiguous" раздражает; выполнение на всех трех catastrophic.

Roots исправляют первое: client declares at `initialize` set of URIs, которых server может касаться. Elicitation исправляет второе: server ставит tool call на паузу и отправляет `elicitation/create`, чтобы попросить пользователя выбрать один вариант.

## Концепция

### Roots

Client объявляет root list в `initialize`:

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

Затем server может вызвать `roots/list`:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Servers ДОЛЖНЫ считать roots границей: любое file read или write вне root set отклоняется. Это не enforced by the client (server все еще code, которому user trusted), но spec-compliant servers это соблюдают.

Когда пользователь добавляет или удаляет root, client отправляет `notifications/roots/list_changed`. Server повторно вызывает `roots/list` и обновляет свою boundary.

### Почему roots — client primitive

Roots объявляются client, потому что они представляют user's consent model. User сказал Claude Desktop: "дай этому notes server доступ к этим двум директориям". Server не может расширить этот scope.

### Elicitation: form-mode по умолчанию

`elicitation/create` принимает form schema плюс natural-language prompt:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

Client отрисовывает form, собирает ответ пользователя и возвращает:

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

Три возможных actions: `accept` (user заполнил форму), `decline` (user закрыл ее), `cancel` (user прервал весь tool call).

Form schemas flat — nested objects не поддерживаются в v1. SDKs обычно reject все сложнее одного уровня.

### Elicitation: URL mode (SEP-1036, experimental)

Новое в 2025-11-25. Вместо schema server отправляет URL:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

Client открывает URL в browser, ждет completion и возвращается, когда пользователь закончил. Полезно для OAuth flows, payment authorization и document signing, где form недостаточна.

Drift-risk note: форма response SEP-1036 все еще settles; одни SDK возвращают callback URL, другие возвращают completion token. Прочитайте release notes своего SDK перед использованием URL mode в production.

### Когда elicitation подходит

- Подтверждение пользователя перед destructive actions (destructive hint + elicitation).
- Disambiguation (выбор одного из N совпадений).
- First-run setup (API keys, directories, preferences).
- OAuth-style flows (URL mode).

### Когда elicitation не подходит

- Заполнение required arguments tool, о которых модель могла спросить в тексте. Используйте обычный re-prompt, а не elicitation dialog.
- High-frequency calls. Elicitation прерывает conversation; не запускайте его внутри loop.
- Все, что server мог бы проверить постфактум. Выполните validation, верните error и дайте модели спросить пользователя текстом.

### Human-in-the-loop bridge

Elicitation plus sampling together enable MCP's "human-in-the-loop" model. Agent loop server может ставить pause либо для user input (elicitation), либо для model reasoning (sampling). Phase 13 · 11 covered sampling; this lesson covers elicitation. Combine them for full mid-loop control.

## Использование

`code/main.py` расширяет notes server:

- Response `roots/list`, который server повторно запрашивает после notifications root-list-changed.
- Tool `notes_delete`, который uses `elicitation/create` для disambiguation, когда подходят несколько notes.
- Tool `notes_setup`, который uses URL-mode elicitation, чтобы открыть first-run config page (simulated).
- Boundary check, который отклоняет operations на URI вне declared roots.

Demo запускает три scenarios: happy path (one match), disambiguation (three matches, elicitation fires), out-of-root-write (rejected).

## Ship It

Этот урок создает `outputs/skill-elicitation-form-designer.md`. По tool, которому может понадобиться user confirmation или disambiguation, skill проектирует elicitation form schema и message template.

## Упражнения

1. Запустите `code/main.py`. Запустите disambiguation path; подтвердите, что simulated user answer направляется обратно в tool.

2. Добавьте новый tool `notes_archive`, которому каждый раз требуется elicitation confirmation (destructive hint). Проверьте UX: как это соотносится с тем, что model re-asking in text?

3. Реализуйте URL-mode elicitation для first-run OAuth flow. Учтите drift risk и добавьте SDK-version guard.

4. Расширьте обработку `roots/list`: когда приходит notification, server должен атомарно перечитать roots и пересканировать open file handles, которые теперь могут оказаться out of scope.

5. Прочитайте discussion thread issue SEP-1036 на GitHub. Найдите один open question, влияющий на то, как servers должны handle URL-mode callbacks.

## Ключевые термины

| Термин | Как говорят | Что это значит на самом деле |
|------|----------------|------------------------|
| Root | "Consent boundary" | URI, которого client разрешил server касаться |
| `roots/list` | "Server asks for scope" | Client возвращает текущий root set |
| `notifications/roots/list_changed` | "User changed scope" | Client сигнализирует, что root set изменился |
| Elicitation | "Ask the user mid-call" | Server-initiated request на structured user input |
| `elicitation/create` | "Метод" | JSON-RPC method для elicitation requests |
| Form mode | "Schema-driven form" | Flat JSON Schema, rendered как form в client UI |
| URL mode | "Browser redirect" | SEP-1036 experimental; открывает URL и ждет |
| `accept` / `decline` / `cancel` | "User response outcomes" | Три branches, которые обрабатывает server |
| Disambiguation | "Pick one" | Common elicitation use case, когда у tool есть N candidates |
| Flat form | "Только top-level properties" | Elicitation schemas не могут nest |

## Дополнительное чтение

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — канонический reference по roots
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — канонический reference по elicitation
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — разбор additions 2025-11-25
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — proposal URL-mode elicitation (experimental, drift-risk)
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
