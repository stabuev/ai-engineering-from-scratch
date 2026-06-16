# Capstone 01 — Terminal-Native Coding Agent

> К 2026 году форма coding agent уже устоялась. TUI-harness, stateful plan, sandboxed tool surface, цикл, который планирует, действует, наблюдает и восстанавливается. Claude Code, Cursor 3 и OpenCode с высоты 50 футов выглядят одинаково. Этот capstone предлагает собрать такой агент от начала до конца — от CLI до pull request — и измерить его относительно mini-swe-agent и Live-SWE-agent на SWE-bench Pro. Вы поймете, почему сложная часть — не вызов модели, а tool loop, sandbox и потолок стоимости на 50-turn run.

**Тип:** Capstone
**Языки:** TypeScript / Bun (harness), Python (eval scripts)
**Предварительные требования:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Задействованные фазы:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Время:** 35 часов

## Цели обучения

- Построить терминального кодинг-агента: TUI-оболочку, состояние плана, песочницу инструментов и цикл план-действие-наблюдение-восстановление.
- Реализовать цикл агента с файловыми и shell-инструментами и восстановлением после ошибок.
- Добавить ограждения и верификацию, чтобы агент безопасно правил реальный репозиторий.

## Проблема

Coding agents стали доминирующей категорией AI-приложений в 2026 году. Claude Code (Anthropic), Cursor 3 with Composer 2 and Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k stars), Factory Droids и Google Jules поставляют вариации одной архитектуры: terminal harness, permissioned tool surface, sandbox и plan-act-observe loop вокруг frontier model. Передний край узкий — Live-SWE-agent достиг 79.2% на SWE-bench Verified с Opus 4.5 — но инженерная область широкая. Большинство отказов — не ошибки модели. Это нестабильность tool loop, context poisoning, runaway token cost и разрушительные операции с filesystem.

Невозможно понять эти агенты снаружи. Нужно собрать свой, увидеть, как loop падает на turn 47, когда ripgrep возвращает 8MB совпадений, и перестроить truncation layer. В этом смысл capstone.

## Концепция

У harness четыре поверхности. **Plan** поддерживает state object в стиле TodoWrite, который модель переписывает на каждом turn. **Act** dispatches tool calls (read, edit, run, search, git). **Observe** захватывает stdout / stderr / exit codes, обрезает и возвращает summary обратно. **Recover** обрабатывает tool errors, не взрывая context window и не зацикливаясь навсегда. Форма 2026 года добавляет еще одну вещь: **hooks**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop` и `PreCompact` — настраиваемые extension points, куда operator внедряет policy, telemetry и guardrails.

Sandbox — это E2B или Daytona. Каждая task запускается в свежем devcontainer с git worktree, смонтированным read-write. Harness никогда не трогает host filesystem. Worktree уничтожается при успехе или ошибке. Cost control enforced на трех уровнях: per-turn token ceiling, per-session dollar budget и hard turn limit (обычно 50). Observability layer — OpenTelemetry spans с GenAI semantic conventions, отправляемые в self-hosted Langfuse.

## Архитектура

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Стек

- Harness runtime: Bun 1.2 + Ink 5 (React-in-terminal)
- Model access: OpenRouter unified API with Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (for hardest tasks)
- Tool transport: Model Context Protocol StreamableHTTP (MCP 2026 revision)
- Sandbox: E2B sandboxes (JS SDK) or Daytona devcontainers
- Code search: ripgrep subprocess, tree-sitter parsers for 17 languages (pre-compiled)
- Isolation: `git worktree add` per task, cleanup on success / failure
- Eval harness: SWE-bench Pro (verified subset) + Terminal-Bench 2.0 + your own 30-task holdout
- Observability: OpenTelemetry SDK with `gen_ai.*` semconv → self-hosted Langfuse
- PR posting: GitHub App with fine-grained token, scope limited to the target repo

## Соберите

1. **TUI and command loop.** Создайте scaffold Bun project с Ink. Принимайте `agent run <repo> "<task>"`. Выводите split view: plan pane (top), tool-call stream (middle), token budget (bottom). Добавьте cancel по Ctrl-C, который запускает hook `SessionEnd` перед выходом.

2. **Plan state.** Определите typed TodoWrite schema (pending / in_progress / done items with notes). Модель переписывает полный state на каждом turn как tool call — не позволяйте ей mutate incrementally. Persist plan в `.agent/state.json`, чтобы после crash можно было resume.

3. **Tool surface.** Определите шесть tools: `read_file`, `edit_file` (with diff preview), `ripgrep`, `tree_sitter_symbols`, `run_shell` (with timeout), `git` (status / diff / commit / push). Expose over MCP StreamableHTTP, чтобы harness был transport-agnostic. Каждый tool возвращает truncated output (cap at 4k tokens per call).

4. **Sandbox wrapping.** Каждая task создает E2B sandbox. `git worktree add -b agent/$TASK_ID` свежую branch. Все tool calls выполняются внутри sandbox. Host filesystem недоступна.

5. **Hooks.** Реализуйте все восемь hook types 2026 года. Подключите как минимум четыре user-authored hooks: (a) `PreToolUse` destructive-command guard, который блокирует `rm -rf` outside the worktree, (b) `PostToolUse` token accounting, (c) `SessionStart` budget initialization, (d) `Stop` writes a final trace bundle.

6. **Eval loop.** Склонируйте 30-issue subset of SWE-bench Pro Python. Запустите harness на каждой. Сравните с mini-swe-agent (minimal baseline) по pass@1, turns-per-task и $-per-task. Запишите результаты в `eval/results.jsonl`.

7. **Cost control.** Hard cutoffs: 50 turns, 200k context, $5 per task. Hook `PreCompact` summarizes older turns into a prior-state block на отметке 150k, освобождая место для новых observations без потери plan.

8. **PR posting.** При успехе final step — `git push` + GitHub API call, который открывает PR с plan и diff summary в body.

## Использование

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Что сдавать

Deliverable skill живет в `outputs/skill-terminal-coding-agent.md`. По repo path и task description он запускает полный plan-act-observe loop в sandbox и возвращает PR URL плюс trace bundle. Rubric для этого capstone:

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Your harness vs mini-swe-agent on 30 matched Python tasks |
| 20 | Architecture clarity | Plan/act/observe separation, hook surface, tool schema — reviewed against Live-SWE-agent layout |
| 20 | Safety | Sandbox escape tests, permission prompts, destructive-command guard passes red-team |
| 20 | Observability | Trace completeness (100% of tool calls spanned), token accounting per turn |
| 15 | Developer UX | Cold-start < 2s, crash recovery resumes plan, Ctrl-C cancels mid-tool cleanly |
| **100** | | |

## Упражнения

1. Замените backing model с Claude Sonnet 4.7 на Qwen3-Coder-30B, served on vLLM. Сравните pass@1 и $-per-task. Сообщите, где open model underperforms.

2. Добавьте `reviewer` sub-agent, который читает diff перед PR posting и может запросить revision loop. Измерьте, снижает ли false-positive reviews SWE-bench pass rate ниже single-agent baseline (hint: usually yes).

3. Stress-test sandbox: напишите task, которая пытается `curl` external URL, и task, которая пишет outside the worktree. Подтвердите, что обе заблокированы PreToolUse hook. Log the attempts.

4. Реализуйте `PreCompact` summarization с smaller model (Haiku 4.5). Измерьте, сколько plan fidelity теряется при 3x compaction.

5. Замените MCP StreamableHTTP transport на stdio. Benchmark cold-start and per-call latency. Выберите winner для local-only use.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Harness | "The agent loop" | Код вокруг модели, который dispatches tools, maintains plan state и enforces budgets |
| Hook | "Agent event listener" | User-authored script, запускаемый harness на одном из восьми lifecycle events |
| Worktree | "Git sandbox" | Linked git checkout по отдельному path; disposable без touching the main clone |
| TodoWrite | "Plan state" | Typed list of pending/in-progress/done items, который модель rewrites each turn |
| StreamableHTTP | "MCP transport" | 2026 MCP revision: long-lived HTTP connection with bidirectional streaming; replaces SSE |
| Token ceiling | "Context budget" | Per-turn или per-session cap на input+output tokens; triggers compaction or termination |
| pass@1 | "Single-attempt pass rate" | Доля SWE-bench tasks, решенных с первого запуска без retry или test-set peeking |

## Дополнительное чтение

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — reference harness from Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs and Composer 2 product notes
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — minimal baseline for SWE-bench harness comparison
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79.2% SWE-bench Verified with Opus 4.5
- [OpenCode](https://opencode.ai) — open harness, 112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — the evaluation this capstone targets
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — span schema for tool calls and token usage
