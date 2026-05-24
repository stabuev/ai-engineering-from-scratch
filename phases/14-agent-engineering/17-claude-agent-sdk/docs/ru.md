# Claude Agent SDK: подагенты и хранилище сессий

> Claude Agent SDK — библиотечная форма harness Claude Code. Built-in tools, subagents для изоляции context, hooks, W3C trace propagation, parity session store. Claude Managed Agents — hosted-альтернатива для долгих async-задач.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 10 (Skill Libraries)
**Время:** ~75 минут

## Цели обучения

- Объяснить разницу между Anthropic Client SDK (raw API) и Claude Agent SDK (форма harness).
- Описать subagents — parallelization и context isolation — и когда к ним обращаться.
- Назвать поверхность session store в Python SDK (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) и роль `--session-mirror`.
- Реализовать stdlib harness с built-in tools, запуском subagent с изолированным context, lifecycle hooks и session store.

## Проблема

Raw LLM API дает один round-trip. Production agent нуждается в tool execution, MCP servers, lifecycle hooks, запуске subagent, session persistence, trace propagation. Claude Agent SDK поставляет эту форму как библиотеку — тот же harness, который использует Claude Code, но открытый для custom agents.

## Концепция

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** Raw Messages API. Вы владеете loop, tools, state.
- **Agent SDK (`claude-agent-sdk`).** Built-in tool execution, MCP connections, hooks, запуск subagent, session store. Loop Claude Code как библиотека.

### Built-in tools

SDK поставляет 10+ tools из коробки: file read/write, shell, grep, glob, web fetch и другие. Custom tools регистрируются через стандартный tool-schema interface.

### Подагенты

Две цели, задокументированные Anthropic:

1. **Parallelization.** Выполнять независимую работу конкурентно. "Найди test file для каждого из этих 20 modules" — это 20 параллельных subagent tasks.
2. **Context isolation.** Subagents используют собственное context window; orchestrator получает только results. Budget orchestrator сохраняется.

Недавние добавления Python SDK: `list_subagents()`, `get_subagent_messages()` для чтения transcripts subagent.

### Session store

Protocol parity с TypeScript:

- `append(session_id, message)` — добавить turn.
- `load(session_id)` — восстановить conversation.
- `list_sessions()` — перечислить.
- `delete(session_id)` — с cascade к subagent sessions.
- `list_subkeys(session_id)` — перечислить subagent keys.

`--session-mirror` (CLI flag) зеркалит transcript во внешний файл по мере stream, для отладки.

### Hooks

Lifecycle hooks, которые можно зарегистрировать:

- `PreToolUse`, `PostToolUse` — gate или audit tool calls.
- `SessionStart`, `SessionEnd` — подготовка и завершение.
- `UserPromptSubmit` — действие над user input до того, как его увидит модель.
- `PreCompact` — запуск перед context compaction.
- `Stop` — cleanup при выходе agent.
- `Notification` — side-channel alerts.

Hooks — это способ, которым pro-workflow (ссылка на curriculum Фазы 14) и похожие системы добавляют cross-cutting behavior.

### W3C trace context

OTel spans, активные у caller, propagates в CLI subprocess через W3C trace context headers. Весь multi-process trace отображается как один trace в вашем backend.

### Claude Managed Agents

Hosted-альтернатива (beta header `managed-agents-2026-04-01`). Долгая async-работа, built-in prompt caching, built-in compaction. Вы обмениваете контроль на managed infrastructure.

### Где этот паттерн ломается

- **Subagent over-spawn.** Запуск 100 subagents для 100 крошечных tasks. Overhead доминирует. Вместо этого batching.
- **Hook creep.** Каждая команда добавляет hooks; startup time раздувается. Пересматривайте hooks ежеквартально.
- **Session bloat.** Sessions накапливаются; размер растет. Используйте `list_sessions` + expiry policy.

## Соберите это

`code/main.py` реализует форму SDK на stdlib:

- `Tool`, `ToolRegistry` со встроенными `read_file`, `write_file`, `list_dir`.
- `Subagent` — private context, isolated run, возвращаемые results.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- Demo: main agent запускает 3 subagents параллельно (каждый изолирован), агрегирует results, сохраняет session.

Запустите:

```
python3 code/main.py
```

Trace показывает context isolation subagent (размер context orchestrator остается ограниченным), выполнение hooks и session persistence.

## Используйте это

- **Claude Agent SDK** для Claude-first продуктов, которым нужна форма harness Claude Code.
- **Claude Managed Agents** для hosted долгой async-работы.
- **OpenAI Agents SDK** (Урок 16) для OpenAI-first аналогов.
- **LangGraph + custom tools**, если нужна state machine в форме graph.

## Отправьте в работу

`outputs/skill-claude-agent-scaffold.md` создает scaffold приложения Claude Agent SDK с subagents, hooks, session store, подключением MCP server и W3C trace propagation.

## Упражнения

1. Добавьте spawner subagent, который группирует 20 tasks в batches по 5 параллельных subagents. Измерьте размер context orchestrator vs one-per-task.
2. Реализуйте hook `PreToolUse`, который rate-limits вызовы `write_file` (5 в минуту на session). Протрассируйте поведение.
3. Подключите `list_subkeys`, чтобы отрисовать дерево subagent. Как выглядит глубокая вложенность?
4. Перенесите toy в настоящий Python package `claude-agent-sdk`. Что меняется в tool registration?
5. Прочитайте документацию Claude Managed Agents. Когда вы бы переключились с self-hosted на managed?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | Форма harness: tools, MCP, hooks, subagents, session store |
| Subagent | "Child agent" | Отдельный context, собственный budget; results поднимаются наверх |
| Session store | "Conversation DB" | Persist, load, list, delete turns с cascade subagent |
| Hook | "Lifecycle callback" | Pre/post tool, session, prompt submit, compact, stop |
| W3C trace context | "Cross-process trace" | Parent span propagates в CLI subprocess |
| Managed Agents | "Hosted harness" | Anthropic-hosted долгая async-работа |
| `--session-mirror` | "Transcript mirror" | Записывает session turns во внешний файл по мере stream |
| MCP server | "Tool surface" | Внешний источник tools/resources, подключенный к agent |

## Дополнительное чтение

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — библиотечная форма Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — production patterns
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted-альтернатива
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — аналог
