# Capstone — построить полную экосистему инструментов

> Фаза 13 научила каждому элементу. Этот capstone соединяет их в одну production-shaped систему: MCP server с tools + resources + prompts + tasks + UI, OAuth 2.1 на edge, RBAC gateway, multi-server client, A2A sub-agent call, OTel tracing в collector, обнаружение tool-poisoning в CI и bundle AGENTS.md + SKILL.md. К концу вы сможете защитить каждое архитектурное решение.

**Тип:** Сборка
**Языки:** Python (stdlib, end-to-end harness экосистемы)
**Предварительные требования:** Фаза 13 · 01-21
**Время:** ~120 минут

## Цели обучения

- Составить MCP server, который выставляет tools, resources, prompts и task с app `ui://`.
- Поставить перед server OAuth 2.1 gateway, который применяет RBAC и pinned hashes.
- Написать multi-server client, который ведет tracing end-to-end с атрибутами OTel GenAI.
- Делегировать часть workload в A2A sub-agent; проверить, что opacity сохранена.
- Упаковать весь stack с AGENTS.md + SKILL.md, чтобы другие агенты могли им управлять.

## Проблема

Отгрузите систему "research and report":

- Пользователь просит: "суммируй три самые цитируемые статьи arXiv 2026 года о протоколах агентов."
- Система: ищет arXiv через MCP; делегирует summarization papers специализированному writer agent через A2A; агрегирует results; рендерит interactive report как MCP Apps resource `ui://`; логирует каждый шаг в OTel.

Все primitives из Фазы 13 появляются здесь. Это не toy — production research-assistant systems, выпущенные в 2026 году Anthropic (Claude Research product), OpenAI (GPTs with Apps SDK) и third parties, имеют именно такую форму.

## Концепция

### Архитектура

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Иерархия trace

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

Один trace id. Каждый span имеет правильные атрибуты `gen_ai.*`.

### Security posture

- OAuth 2.1 + PKCE с resource indicator и привязкой audience к gateway.
- Gateway хранит upstream credentials; user их никогда не видит.
- RBAC: `alice` имеет `research:read`, `research:write`, может вызывать все tools. `bob` имеет `research:read`, не может вызвать `generate_report`.
- Pinned description manifest: отклоняется любой server, у которого изменились hashes tools.
- Rule of Two audit: ни один tool не объединяет untrusted input, sensitive data и consequential action.

### Rendering

Финальная task `generate_report` возвращает content blocks плюс resource `ui://report/current`. Host клиента (Claude Desktop и т. д.) рендерит interactive dashboard в sandbox iframe. Dashboard содержит отсортированный список papers, citation counts и button, который вызывает `host.callTool('summarize_paper', {arxiv_id})` для любой paper, по которой user кликает.

### Packaging

Все это поставляется как:

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

Пользователи разворачивают систему командой `docker compose up`. Пользователи Claude Code, Cursor, Codex и opencode могут управлять системой, вызывая skill `run-research`.

### Что внес каждый урок Фазы 13

| Урок | Что использует capstone |
|--------|------------------------|
| 01-05 | Интерфейс tools, provider-portability, parallel calls, schemas, linting |
| 06-10 | MCP primitives, server, client, transports, resources + prompts |
| 11-14 | Sampling, roots + elicitation, async tasks, apps `ui://` |
| 15-17 | Tool poisoning, OAuth 2.1, gateway + registry |
| 18 | Делегирование A2A sub-agent |
| 19 | Tracing OTel GenAI |
| 20 | Routing gateway для LLM layer |
| 21 | SKILL.md + AGENTS.md packaging |

## Используйте

`code/main.py` сшивает patterns из предыдущих уроков в одно runnable demo. Все на stdlib, все in-process, чтобы можно было прочитать end to end. Он запускает полный flow для сценария research-and-report: handshake with gateway, симуляция OAuth 2.1, объединенный `tools/list`, `generate_report` как task, A2A call к writer, возврат resource `ui://`, испускание OTel spans.

На что обратить внимание:

- Один trace id через каждый hop.
- Policy gateway блокирует запись второму user.
- Жизненный цикл Task идет working → completed и возвращает и text, и content `ui://`.
- Внутреннее состояние A2A call непрозрачно для orchestrator.
- AGENTS.md и SKILL.md — единственные files, которые нужны другому agent для воспроизведения workflow.

## Отгрузите

Этот урок создает `outputs/skill-ecosystem-blueprint.md`. Для product need (research, summarization, automation) skill создает full architecture: какие MCP primitives, какие gateway controls, какие A2A calls, какая telemetry, какая packaging.

## Упражнения

1. Запустите `code/main.py`. Отметьте единый trace id и то, как spans вкладываются друг в друга. Посчитайте, сколько primitives из Фазы 13 затрагивает demo.

2. Расширьте demo: добавьте второй backend MCP server (например, `bibliography`) и подтвердите, что gateway объединяет его tools в тот же namespace.

3. Замените fake A2A writer agent на реальный, running on a subprocess. Используйте harness из урока 19.

4. Добавьте шаг PII redaction в routing gateway между orchestrator и LLM. Подтвердите, что emails в user query очищаются.

5. Напишите AGENTS.md для teammate, который будет сопровождать эту систему. Он должен читаться менее чем за пять минут и давать все, что нужно, чтобы управлять capstone в Cursor или Codex.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Capstone | "Интеграционное demo Фазы 13" | End-to-end система, использующая каждый primitive |
| Research and report | "Сценарий" | Pattern search, summarize, render |
| Ecosystem | "Все части вместе" | Server + client + gateway + sub-agent + telemetry + package |
| Trace hierarchy | "Единый trace id" | Span каждого hop делит trace; parent-child через span ids |
| Gateway-issued token | "Транзитивная auth" | Client видит только token gateway; gateway хранит upstream creds |
| Merged namespace | "Все tools в одном плоском списке" | Multi-server merge на gateway, prefix-on-collision |
| Opacity boundary | "A2A call скрывает internals" | Рассуждение sub-agent невидимо orchestrator |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | Project context + workflow + tools |
| Defense-in-depth | "Несколько security layers" | Pinned hashes, OAuth, RBAC, Rule of Two, audit log |
| Spec compliance matrix | "Что мы отгружаем из требуемого спецификацией" | Checklist, mapping deliverables to требованиям 2025-11-25 |

## Дополнительное чтение

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — сводный reference
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — куда движется protocol
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — reference A2A v1.0
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — канонические tracing conventions
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — patterns production agent runtime
