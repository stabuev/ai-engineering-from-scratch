# MCP Auth in Production — DCR, JWKS Rotation, Audience-Pinned Tokens на iii Primitives

> Lesson 16 поднял OAuth 2.1 state machine in memory. К 2026 каждый MCP server, который вы поставляете в реальную организацию, находится за production auth: dynamic client registration (RFC 7591), discovery metadata authorization server (RFC 8414), JWKS rotation, которая не ломает token validation в 3 a.m., и audience-pinned tokens, которые отказываются от confused-deputy reuse. Этот урок проводит все это через iii primitives — `iii.registerTrigger` для HTTP и cron, `iii.registerFunction` для auth logic, `state::set/get` для cached keys — так что auth surface становится observable, restartable и replayable, как любая другая workload в engine.

**Тип:** Build
**Языки:** Python (stdlib, iii primitives, замоканные для среды урока)
**Предварительные требования:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Время:** ~90 минут

## Цели обучения

- Найти authorization server через RFC 8414 metadata и проверить contract.
- Реализовать RFC 7591 dynamic client registration, чтобы MCP clients регистрировались без admin intervention.
- Кэшировать и rotate JWKS keys с cron trigger, чтобы signature verification переживала key roll-over.
- Pin tokens к одному MCP resource через RFC 8707 resource indicators и refuse confused-deputy reuse.
- Подключить каждый endpoint и background job как iii primitives — HTTP triggers, cron triggers, named functions и `state::*` reads — чтобы один restart rebuilds auth surface.
- Читать IdP capability matrix и refuse to deploy, когда IdP не может satisfy MCP auth profile.

## Проблема

Simulator Lesson 16 запускает OAuth 2.1 in memory. Production имеет три operational gaps, которых memory-only simulator не видит.

Первый gap — enrollment. Реальная организация запускает сотни MCP servers и тысячи MCP clients. Operators не регистрируют каждого пользователя Cursor вручную как OAuth client. RFC 7591 dynamic client registration позволяет client сделать `POST /register` к authorization server и на месте получить `client_id` (и опционально `client_secret`). Server публикует `registration_endpoint` в RFC 8414 metadata; client discovers it без out-of-band configuration.

Второй gap — key rotation. JWT validation зависит от signing keys authorization server, опубликованных как JSON Web Key Set (JWKS). Authorization server rotates these on a schedule (часто каждый час, иногда быстрее при incident response). MCP server, который fetches JWKS once at boot, validates нормально до rotation window — затем каждый request fails until restart. Production wires JWKS как cached value с refresh job, который перезаписывает cache до истечения предыдущих keys, плюс fall-back fetch on cache miss для случая, когда приходит token, подписанный key newer than cache.

Третий gap — audience binding. Lesson 16 ввел RFC 8707 resource indicators. В production этот indicator становится жесткой проверкой claim на каждом request. MCP server сравнивает `token.aud` со своим canonical resource URL и отклоняет mismatches с HTTP 401. Это единственная защита от upstream MCP server (или malicious client, holding token meant for one server), который replays that token against another server in the same trust mesh.

Этот урок рассматривает каждый из этих gaps как iii primitive. Metadata document — HTTP trigger, возвращающий output function. JWKS rotation — cron trigger, вызывающий `auth::rotate-jwks`, который пишет в `state::set("auth/jwks/<issuer>", ...)`. JWT validation — function, которую другие вызывают через `iii.trigger("auth::validate-jwt", token)`. Сам MCP server — просто еще один HTTP trigger, который calls into validation before dispatching. Перезапуск engine: trigger registry rebuilds; state survives; auth surface operational without manual reconciliation.

## Концепция

### RFC 8414 — OAuth Authorization Server Metadata

Документ по адресу `/.well-known/oauth-authorization-server` описывает все, что нужно client:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Client, которому дан MCP resource URL, выполняет цепочку discovery: `oauth-protected-resource` из RFC 9728 (документ resource server) называет issuer, затем `oauth-authorization-server` (этот RFC) называет every endpoint. Client никогда не hard-codes authorization URL.

Контракт, который нужно проверить перед доверием IdP для MCP:

- `code_challenge_methods_supported` включает `S256` (PKCE по RFC 7636).
- `grant_types_supported` включает `authorization_code` и отклоняет `password` и `implicit`.
- `registration_endpoint` присутствует (поддержка RFC 7591).
- `response_types_supported` равен ровно `["code"]` для OAuth 2.1.

Если чего-то нет, MCP server отказывается разворачиваться с этим IdP. Ошибка в deployment manifest, а не в коде.

### RFC 9728 (recap) — Protected Resource Metadata

Lesson 16 covered RFC 9728. Отличие в production: этот документ — единственное место, где client ищет authorization servers, которым доверяет *этот* MCP server. Один MCP server может принимать tokens от нескольких IdPs (один для staff, один для partners). RFC 9728 объявляет этот набор; RFC 8414 документирует, что поддерживает каждый IdP.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — Dynamic Client Registration

Без DCR каждый MCP client (Cursor, Claude Desktop, custom agent) требует out-of-band exchange с IdP admin. С DCR client отправляет:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server отвечает `client_id` и `registration_access_token` для последующих updates:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` — правильный default для MCP clients, running on user's device. Они получают только `client_id` — нет `client_secret`, который можно exfiltrate. PKCE provides proof-of-possession, который нужен public clients.

Три production pitfalls:

- Registration endpoint должен rate-limit по source IP. Без этого hostile actor scripts millions of fake registrations и exhausts `client_id` namespace. iii makes this trivial: registration HTTP trigger calls `auth::rate-limit` function before dispatching to registrar.
- `software_statement` (signed JWT, vouching for the client) required by some enterprise IdPs. Lesson mock skips it; production wires verification step, который rejects unsigned registrations from anything other than localhost redirect URIs.
- `registration_access_token` должен храниться как hash, а не plaintext. Кража этого token означает, что attacker can rewrite client's redirect URIs.

### RFC 8707 (recap) — Resource Indicators

Lesson 16 established the shape. Production rule: каждый token request includes `resource=<canonical-mcp-url>`, и MCP server verifies `token.aud` matches its own resource URL on every call. Если MCP server reachable at `https://notes.example.com/mcp`, canonical URL is `https://notes.example.com` — path component excluded, so one server hosts multiple paths under one audience.

### RFC 7636 (recap) — PKCE

PKCE mandatory in OAuth 2.1. Authorization-code flow урока always carries `code_challenge` and `code_verifier`. Server rejects any token request without verifier or with verifier that does not hash to stored challenge.

### MCP Spec 2025-11-25 Auth Profile

MCP spec (2025-11-25) точно описывает, что должен делать authorization layer MCP server:

- Publish `/.well-known/oauth-protected-resource` (RFC 9728).
- Принимать tokens только через `Authorization: Bearer ...`.
- Проверять `aud`, `iss`, `exp` и required scopes на каждом request.
- Отвечать `WWW-Authenticate` с `Bearer error=...` для каждого 401 и 403, включая параметры `scope=` и `resource=`, где это применимо.
- Отклонять tokens, у которых `aud` не совпадает с canonical resource.
- Отклонять tokens, у которых `iss` отсутствует в списке `authorization_servers` из protected-resource metadata.

OAuth 2.1 draft — substrate; RFC 8414/7591/8707/9728 + RFC 7636 — surface; MCP spec — profile.

### IdP capability matrix

Не каждый IdP supports full MCP profile. Матрица ниже документирует фактические capabilities по состоянию на спецификацию 2025-11-25. Это *deployment gate*, а не recommendation.

| Категория IdP | RFC 8414 metadata | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | Примечания |
|---|---|---|---|---|---|
| Self-hosted (Keycloak) | yes | yes | yes (since 24.x) | yes | Reference IdP для MCP profile в этом уроке; поддерживает каждый RFC end-to-end. |
| Enterprise SSO (Microsoft Entra ID) | yes | yes (premium tiers) | yes | yes | Доступность DCR зависит от tier tenant; verify in target tenant before deploying. |
| Enterprise SSO (Okta) | yes | yes (Okta CIC / Auth0) | yes | yes | DCR доступен в Auth0 (теперь Okta CIC); classic Okta orgs требуют admin pre-registration. |
| Social login IdPs (generic) | varies | rarely | rarely | yes | Большинство social IdPs treat clients as static partners; не rely on DCR. Используйте как identity source only, layer your own MCP-aware authorization server on top. |
| Custom / homegrown | depends | depends | depends | depends | Если поставляете свой вариант, ship the full profile. Пропуск любого из четырех RFC выше ломает MCP auth contract. |

Refusal rule для deployment manifest: если выбранный IdP не возвращает `registration_endpoint` и не перечисляет `S256` в `code_challenge_methods_supported`, MCP server отказывается стартовать. Degraded mode отсутствует.

### JWKS rotation pattern with iii

Production failure mode — stale JWKS cache. Решение: cron trigger и cache `state::*`:

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

Каждые шесть часов cron trigger вызывает `auth::rotate-jwks`, который fetches `<issuer>/.well-known/jwks.json` and writes to `state::set("auth/jwks/<issuer>", {keys, fetched_at})`. Validator reads from `state::get`. Token, whose `kid` missing from cache, triggers synchronous `auth::rotate-jwks` call as fall-back. Это одновременно обрабатывает scheduled rotation (cron) и key-overlap windows (synchronous fall-back).

Форма состояния:

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Два ключа одновременно — steady state. Authorization servers rotate by introducing next key (`k_2026_04`) before retiring previous (`k_2026_03`), so tokens issued under old key remain valid until they expire. Cache holds the union; validator picks by `kid`.

### Подключение iii primitives (та часть, ради которой нужен этот урок)

Пять primitives compose auth surface:

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

Сам MCP server никогда не вызывает validation напрямую. Он делает:

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

Эта косвенность — ставка iii. Завтра вы замените validator на fanout, который consults two IdPs in parallel, или добавите span emitter, или cache positive validations. MCP server does not change.

### Confused-deputy walkthrough with audience binding

Server A (`notes.example.com`) и Server B (`tasks.example.com`) оба register against same authorization server. Server A compromised. Attacker takes user's notes token and replays it against Server B.

Validator Server B:

1. Decode JWT, fetch JWKS by `kid`, verify signature.
2. Check `iss` against its protected-resource metadata's `authorization_servers`. (Pass — same IdP.)
3. Check `aud == "https://tasks.example.com"`. (Fail — token's `aud` is `https://notes.example.com`.)
4. Return 401 with `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`.

Audience claim — единственная protocol-layer защита от этой атаки. Skipping it for performance — самая частая production mistake; validator must run on every request, not just at session start.

### Failure modes

- **Stale JWKS.** Validator rejects valid tokens after key rotation. Fix — cron+fall-back pattern above. Never cache JWKS without refresh job.
- **Missing `aud` claim.** Some IdPs default to omitting `aud`, unless `resource` present in token request. Validator must reject tokens with missing `aud`, not treat absence as wildcard.
- **Scope upgrade race.** Two concurrent step-up flows for same user can both succeed and produce two access tokens with different scopes. Validator must use token presented on request, not look up "the user's current scope" — that creates TOCTOU window.
- **Registration token theft.** Leaked `registration_access_token` lets attacker rewrite redirect URIs. Hash these at rest; require client to present cleartext on every update; rotate on suspicion.
- **`iss` not pinned.** Validator, accepting any `iss`, lets attacker stand up their own authorization server, register client for target audience and issue tokens. Protected-resource metadata's `authorization_servers` list is allow-list; enforce it.

## Использование

`code/main.py` проходит полный production flow на stdlib Python и небольшом registry `iii_mock`, который mimics `iii.registerFunction`, `iii.registerTrigger`, `iii.trigger` and `state::set/get`. Flow:

1. Authorization server публикует RFC 8414 metadata at `/.well-known/oauth-authorization-server`.
2. MCP client вызывает metadata endpoint и discovers registration endpoint.
3. MCP client отправляет POST на `/register` (RFC 7591) и получает `client_id`.
4. MCP client выполняет PKCE-protected authorization code flow (RFC 7636) с `resource` indicator (RFC 8707).
5. MCP client вызывает tool на MCP server с `Authorization: Bearer ...`.
6. MCP server triggers `auth::validate-jwt`, который reads JWKS from `state::get`.
7. Cron trigger fires `auth::rotate-jwks`, replacing JWKS in state.
8. Следующий call validates against new keys without restart.
9. Попытка confused-deputy против другого MCP resource получает 401 с audience mismatch.

Mock JWT здесь uses HS256 with shared secret (so lesson runs on stdlib only). Production uses RS256 или EdDSA с JWKS pattern выше; validation logic otherwise identical.

## Результат

Этот урок создает `outputs/skill-mcp-auth-iii.md`. Для MCP server config и IdP capability set skill emits iii primitives to register, JWKS rotation schedule, scope mapping и refusal rules, применяемые, когда IdP не поддерживает полный RFC profile.

## Упражнения

1. Запустите `code/main.py`. Проследите 9-step flow. Отметьте, где `state::get` возвращает stale data immediately before `auth::rotate-jwks` overwrites it, и как следующий request validates against new key.

2. Добавьте новый IdP в список `authorization_servers` protected-resource metadata. Issue token, signed by new IdP, и подтвердите, что validator accepts it. Issue token, signed by unlisted IdP, и подтвердите, что validator rejects with `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. Реализуйте `auth::rate-limit` как iii function и вызовите ее из registration HTTP trigger before registrar runs. Используйте token-bucket per source IP, held in `state::set("auth/ratelimit/<ip>", ...)`.

4. Прочитайте RFC 7591 и определите два поля, которые lesson's `/register` handler does not validate. Добавьте validation. (Hint: `software_statement` and `redirect_uris` URI scheme.)

5. Прочитайте authorization section MCP spec 2025-11-25. Найдите одно normative requirement к headers `WWW-Authenticate`, которое lesson's validator currently does not emit. Добавьте его.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| ASM | "OAuth metadata document" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "Self-service client registration" | RFC 7591 `POST /register` flow |
| JWKS | "Public keys for JWT validation" | JSON Web Key Set, fetched from `jwks_uri`, indexed by `kid` |
| Resource indicator | "Audience parameter" | Параметр RFC 8707 `resource`, закрепляющий token за одним server |
| `aud` claim | "Audience" | JWT claim, который validator compares against canonical resource URL |
| Confused deputy | "Token replay" | Attack, где token issued for Server A presented to Server B |
| `iss` allow-list | "Trusted authorization servers" | Set, named in protected-resource metadata's `authorization_servers` |
| Key rotation | "Rolling JWKS" | Periodic replacement of signing keys with overlap windows |
| Public client | "Native or browser client" | OAuth client with no `client_secret`; PKCE compensates |
| `WWW-Authenticate` | "401/403 response header" | Carries `Bearer error=...` directives that drive client recovery |

## Дополнительное чтение

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP auth profile, реализованный в этом уроке
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — public-client proof-of-possession
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — resource server discovery
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — consolidated OAuth substrate
