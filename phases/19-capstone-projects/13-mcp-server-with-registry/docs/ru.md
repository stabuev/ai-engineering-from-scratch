# Capstone 13 — MCP Server with Registry and Governance

> Model Context Protocol перестал быть будущим и в 2026 году стал стандартной спецификацией для использования инструментов. Anthropic, OpenAI, Google и все крупные IDE поставляют MCP-клиенты. Pinterest опубликовала свою внутреннюю экосистему MCP-серверов. AAIF Registry формализовал метаданные возможностей в `.well-known`. AWS ECS опубликовал эталонное stateless-развертывание. Block `goose-agent` встроил тот же протокол в hosted assistant. Производственный контур 2026 года выглядит так: StreamableHTTP transport, OAuth 2.1 scopes, OPA policy gating и registry, который позволяет платформенным командам находить, валидировать и включать серверы. Собери это end to end.

**Тип:** Capstone
**Языки:** Python (server, via FastMCP) или TypeScript (@modelcontextprotocol/sdk), Go (registry service)
**Пререквизиты:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Отрабатываемые фазы:** P11 · P13 · P14 · P17 · P18
**Время:** 25 часов

## Цели обучения

- Построить продакшен MCP-сервер плюс реестр и слой governance.
- Реализовать tools, resources и prompts с авторизацией и реестром обнаружения.
- Добавить политики над тем, какие инструменты клиенты могут подключать.

## Задача

MCP стал lingua franca для использования инструментов. Claude Code, Cursor 3, Amp, OpenCode, Gemini CLI и каждый managed agent теперь потребляют MCP-серверы. Производственные сложности не в написании серверов (FastMCP делает это простым), а в их масштабном развертывании с enterprise-требованиями: OAuth scopes на tenant, OPA policy для destructive tools, stateless-масштабирование StreamableHTTP, registry для discovery, audit logs по каждому tool call. Внутренняя MCP-экосистема Pinterest и спецификация AAIF Registry задают планку 2026 года.

Ты построишь MCP-сервер, раскрывающий 10 внутренних инструментов (Postgres read-only, S3 listing, Jira, Linear, Datadog и т. д.), registry UI для platform discovery и human-approval gate для destructive tools. Нагрузочный тест демонстрирует горизонтальное масштабирование StreamableHTTP. Audit trail должен пройти enterprise security review.

## Концепция

Ревизия MCP 2026 требует StreamableHTTP как transport по умолчанию. В отличие от более ранней схемы stdio-and-SSE, StreamableHTTP по умолчанию stateless: единый HTTP endpoint принимает JSON-RPC requests, стримит responses и поддерживает long-lived connections для notifications. Stateless означает горизонтальное масштабирование за load balancer.

Авторизация строится на OAuth 2.1 с per-tool scopes. Token несет scopes вроде `jira:read`, `s3:list`, `postgres:query:readonly`. MCP server проверяет scopes во время tool-call, а не только при старте session. Для high-risk tools сервер отклоняет любой call, scope которого не повышен до `approved:by:human` за последние N минут — это повышение приходит из Slack review card.

Registry — отдельный service. Каждый MCP server раскрывает документ `.well-known/mcp-capabilities` со своим tool manifest, transport URL и auth requirements. Registry опрашивает, валидирует и индексирует. Platform teams используют registry UI, чтобы видеть, какие tools доступны, какие scopes нужны и какие teams ими владеют.

## Архитектура

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## Стек

- Server framework: FastMCP (Python) или `@modelcontextprotocol/sdk` (TypeScript)
- Transport: StreamableHTTP over HTTPS (stateless)
- Auth: OAuth 2.1 с workload identity через SPIFFE / SPIRE
- Policy: OPA / Rego rules для каждого tool; policy decision service на каждый request
- Registry: self-hosted, потребляет `.well-known/mcp-capabilities` manifests
- Human approval: интерактивное сообщение Slack для destructive tools
- Deployment: AWS ECS Fargate или Fly.io, one server per tenant or shared with tenant scoping
- Audit: structured JSONL в per-tenant bucket с per-call lineage

## Сборка

1. **Поверхность инструментов.** Раскрой 10 внутренних tools: Postgres read-only query, S3 list objects, Jira search/fetch, Linear search/fetch, Datadog metric query, PagerDuty on-call lookup, GitHub read-only, Notion search, Slack search, Salesforce read. У каждого tool есть typed schema и scope label.

2. **FastMCP server.** Подключи tools. Настрой StreamableHTTP transport. Добавь middleware для OAuth token introspection и scope enforcement.

3. **OPA policy.** Rego policy для каждого tool: какие scopes разрешают invocation, какое PII redaction применяется, какие payload-size caps действуют. Decision service вызывается на каждый tool call.

4. **Registry service.** Отдельный Go или TS service, который опрашивает `.well-known/mcp-capabilities` у registered servers, валидирует через JSON Schema и предоставляет list / search / validate / enable-disable UI.

5. **Capability manifest.** Каждый server раскрывает `.well-known/mcp-capabilities` с: tool list, auth requirements, transport URL, owner team, SLO.

6. **Разделение destructive tools.** Tools, которые меняют state (Jira create, Linear create, Postgres write), живут на втором MCP server с более строгим auth flow: tokens должны иметь scope `approved:by:human`, повышенный через Slack card в пределах 15 минут.

7. **Audit log.** Append-only JSONL per tenant: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`. PII redaction через Presidio перед записью.

8. **Нагрузочный тест.** 100 concurrent clients на StreamableHTTP. Продемонстрируй горизонтальное масштабирование добавлением второй replica; покажи, что load balancer перераспределяет нагрузку без session stickiness.

9. **Conformance tests.** Запусти official MCP conformance suite против обоих servers. Пройди все mandatory sections.

## Использование

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## Что сдать

`outputs/skill-mcp-server.md` описывает deliverable. Production-grade MCP server + registry + audit layer для внутренних tools с OAuth 2.1 scopes и OPA gating.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Соответствие спецификации | StreamableHTTP + capability manifest проходит MCP conformance tests |
| 20 | Безопасность | Scope enforcement, OPA coverage по каждому tool, secret hygiene |
| 20 | Наблюдаемость | Per-tool-call audit log с PII redaction |
| 20 | Масштабирование | Демонстрация horizontal scale в нагрузочном тесте на 100 clients |
| 15 | Registry UX | Workflow поиска, валидации и включения/отключения |
| **100** | | |

## Упражнения

1. Добавь новый tool (Confluence search). Проведи его через registry validation flow, не трогая core server.

2. Напиши OPA policy, которая редактирует результаты Postgres query, содержащие колонки с именами `email`, `ssn` или `phone`. Проверь probe query.

3. Сравни StreamableHTTP и stdio по локальной latency. Отчитай per-call p50/p95.

4. Реализуй per-tenant quota: максимум N calls per minute per tool per tenant. Применяй через второе OPA rule.

5. Запусти MCP conformance suite из [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) и исправь каждый failure.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|------------------------|
| StreamableHTTP | "2026 MCP transport" | Stateless HTTP + streaming; заменяет SSE + stdio для networked servers |
| Capability manifest | "Well-known doc" | `.well-known/mcp-capabilities` с tool list, auth, transport URL |
| OPA / Rego | "Policy engine" | Open Policy Agent для авторизации tool calls по внешним rules |
| Scope elevation | "Approved-by-human" | Краткоживущий scope, выдаваемый через Slack approval и обязательный для destructive tools |
| Registry | "Tool discovery" | Service, индексирующий MCP servers по их capability manifests |
| Workload identity | "SPIFFE / SPIRE" | Cryptographic service identity для выпуска OAuth token |
| Conformance suite | "Spec tests" | Official MCP test battery для корректности StreamableHTTP + tool manifest |

## Дополнительное чтение

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata и registry
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — спецификация registry 2026 года
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — эталонное production deployment
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — эталонное internal deployment
- [Block `goose` MCP usage](https://block.github.io/goose/) — эталонный паттерн agent consumption
- [FastMCP](https://github.com/jlowin/fastmcp) — server framework на Python
- [Open Policy Agent](https://www.openpolicyagent.org/) — справочник по policy engine
- [SPIFFE / SPIRE](https://spiffe.io) — справочник по workload identity
