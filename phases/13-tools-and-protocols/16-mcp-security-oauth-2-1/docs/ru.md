# MCP Security II — OAuth 2.1, Resource Indicators, Incremental Scopes

> Удаленным MCP servers нужна authorization, а не только authentication. Спецификация 2025-11-25 согласуется с OAuth 2.1 + PKCE + resource indicators (RFC 8707) + protected-resource metadata (RFC 9728). SEP-835 добавляет incremental scope consent со step-up authorization через 403 WWW-Authenticate. Этот урок реализует step-up flow как конечный автомат, чтобы вы видели каждый переход.

**Тип:** Build
**Языки:** Python (stdlib, симулятор конечного автомата OAuth)
**Предварительные требования:** Phase 13 · 09 (transports), Phase 13 · 15 (security I)
**Время:** ~75 минут

## Цели обучения

- Различать обязанности resource server и authorization server.
- Проследить OAuth 2.1 authorization code flow, защищенный PKCE.
- Использовать `resource` (RFC 8707) и protected-resource metadata (RFC 9728), чтобы предотвращать confused-deputy attacks.
- Реализовать step-up authorization: server отвечает 403 с WWW-Authenticate, запрашивая более высокий scope; client снова запрашивает consent пользователя и повторяет запрос.

## Проблема

Ранний MCP (до 2025) поставлял remote servers с ad-hoc API keys или вообще без auth. Спецификация 2025-11-25 закрывает этот пробел полным профилем OAuth 2.1.

Три реальные потребности:

- **Обычные remote servers.** Пользователь устанавливает remote MCP server, который получает доступ к его Notion / GitHub / Gmail. OAuth 2.1 с PKCE имеет правильную форму.
- **Scope escalation.** Notes server с grant `notes:read` позже может потребовать `notes:write` для конкретного действия. Вместо повторения всего flow step-up (SEP-835) запрашивает дополнительный scope.
- **Confused deputy prevention.** Client держит token, scoped по audience для Server A. Server A malicious и пытается предъявить token Server B. Resource indicators (RFC 8707) pin token к intended audience.

OAuth 2.1 не новый. Новое — профиль MCP: конкретные обязательные flows (только authorization code + PKCE; без implicit, без client credentials по умолчанию), обязательные resource indicators в каждом token request и опубликованные protected-resource metadata, чтобы clients знали, куда идти.

## Концепция

### Роли

- **Client.** MCP client (Claude Desktop, Cursor и т. д.).
- **Resource server.** MCP server (notes, GitHub, Postgres и т. д.).
- **Authorization server.** Выдает tokens. Может быть тем же сервисом, что resource server, или отдельным IdP (Auth0, Keycloak, Cognito).

В профиле MCP resource и authorization servers могут быть одним host, но должны различаться URL.

### Authorization code + PKCE

Flow:

1. Client генерирует `code_verifier` (случайный) и `code_challenge` (SHA256).
2. Client перенаправляет пользователя на `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`.
3. User дает consent. Authorization server перенаправляет на `redirect_uri?code=...`.
4. Client отправляет POST на `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`.
5. Authorization server проверяет hash verifier against stored challenge и выдает access token.
6. Client использует token: `Authorization: Bearer ...` на каждом request к resource server.

PKCE предотвращает authorization-code interception attacks. Resource indicators предотвращают валидность token где-либо еще.

### Protected-resource metadata (RFC 9728)

Resource server публикует документ `.well-known/oauth-protected-resource`:

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Client discovers authorization server from resource server. Это уменьшает configuration — клиенту нужен только resource URL.

### Resource indicators (RFC 8707)

Параметр `resource` в token request закрепляет intended audience token. Выданный token содержит `aud: "https://notes.example.com"`. Другой MCP server, получивший этот token, проверяет `aud` и отклоняет его.

### Scope model

Scopes — это строки, разделенные пробелами. Распространенные MCP conventions:

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` для admin capabilities (используйте осторожно)
- `profile:read` для identity

Выбор scope должен быть least-privilege: запрашивайте то, что нужно сейчас, и делайте step-up, когда понадобится больше.

### Step-up authorization (SEP-835)

Пользователь grants `notes:read`. Позже он просит агента удалить заметку. Сервер отвечает:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Client видит error insufficient_scope, показывает пользователю consent dialog для дополнительного scope, выполняет мини-flow OAuth для него и повторяет request с новым token.

### Token audience validation

Каждый request: server checks `token.aud == self.resource_url`. Mismatch = 401. Это останавливает cross-server token reuse.

### Short-lived tokens и rotation

Access tokens должны быть short-lived (1 час по умолчанию). Refresh tokens rotate on every refresh. Client handles silent refresh in the background.

### No token passthrough

Sampling servers (Phase 13 · 11) не должны передавать token клиента дальше в другие сервисы. Sampling request является boundary.

### Confused deputy prevention

Token привязан к `aud`. Client привязан к `client_id`. Каждый request проверяется по обоим значениям. Спецификация явно запрещает старый паттерн "pass-the-token", распространенный в remote tool ecosystems до MCP.

### Client ID discovery

Каждый MCP client публикует свои metadata по fixed URL. Authorization servers могут fetch client metadata document, чтобы discover redirect URIs и contact info. Это убирает manual client registration.

### Gateways and OAuth

Phase 13 · 17 показывает, как enterprise gateway handles OAuth: gateway хранит credentials для upstream servers, tokens для client выпускает сам gateway, а upstream tokens никогда не покидают gateway. Это переворачивает trust model — users authenticate with the gateway once; gateway handles N server authorizations.

## Использование

`code/main.py` симулирует полный OAuth 2.1 step-up flow как конечный автомат. Он реализует:

- PKCE code-verifier / challenge generation.
- Authorization code flow with resource indicator.
- Protected-resource metadata endpoint.
- Token validation with audience check.
- Step-up on `insufficient_scope`.

В этом уроке нет HTTP server; конечный автомат выполняется in memory, чтобы вы могли проследить каждый hop. Gateway lesson Phase 13 · 17 подключает это к actual transport.

## Результат

Этот урок создает `outputs/skill-oauth-scope-planner.md`. Для remote MCP server с tools skill проектирует набор scopes, правила pinning и step-up policy.

## Упражнения

1. Запустите `code/main.py`. Проследите two-scope step-up flow. Отметьте, какие hops повторяются на step-up.

2. Добавьте refresh-token rotation: каждый refresh выдает новый refresh token и invalidates old one. Симулируйте использование stolen refresh token после rotation и подтвердите, что оно завершается отказом.

3. Реализуйте protected-resource metadata endpoint как настоящий HTTP response с помощью stdlib http.server. Отразите endpoint `/mcp` из Lesson 09.

4. Спроектируйте scope hierarchy для GitHub MCP server: read repo, write PR, approve PR, merge PR, admin. Используйте step-up между каждым уровнем.

5. Прочитайте RFC 8707 и RFC 9728. Определите одно поле в 9728, которое MCP использует иначе, чем example из RFC. (Hint: это касается `scopes_supported`.)

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| OAuth 2.1 | "Modern OAuth" | Consolidated RFC, который mandates PKCE и forbids implicit flow |
| PKCE | "Proof-of-possession" | Code verifier + challenge, defeating authorization-code interception |
| Resource indicator | "Token audience" | Параметр RFC 8707 `resource`, закрепляющий token за одним server |
| Protected-resource metadata | "Discovery doc" | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | "Incremental consent" | Flow SEP-835 для добавления scopes on demand |
| `insufficient_scope` | "403 with WWW-Authenticate" | Сигнал server для re-consent на larger scope |
| Confused deputy | "Token reuse across services" | Attack, где trusted holder неправомерно forwards token |
| Short-lived token | "Access token TTL" | Bearer, который быстро expires; refresh token renews |
| Scope hierarchy | "Least privilege stack" | Graduated scope set со step-up между уровнями |
| Client ID metadata | "Client discovery doc" | URL, по которому client публикует собственные OAuth metadata |

## Дополнительное чтение

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — канонический MCP OAuth profile
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — walkthrough изменений 2025-11-25
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — RFC для audience-pinning
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — RFC discovery-document
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — practical step-up-flow walkthrough
