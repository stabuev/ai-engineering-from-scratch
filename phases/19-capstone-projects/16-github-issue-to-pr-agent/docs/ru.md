# Capstone 16 — GitHub Issue-to-PR Autonomous Agent

> AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud и Google Jules поставляют один и тот же product shape 2026 года: пометь issue label, получи PR. Запусти agent в cloud sandbox, проверь, что tests проходят, и опубликуй review-ready PR с rationale. Сложные части — автоматическое воспроизведение build environment репозитория, предотвращение credential leakage, enforcement per-repo budgets и гарантия, что agent не может force-push. Этот capstone строит self-hosted version и сравнивает ее по cost и pass rate с hosted alternatives.

**Тип:** Capstone
**Языки:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Пререквизиты:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Отрабатываемые фазы:** P11 · P13 · P14 · P15 · P17
**Время:** 30 часов

## Цели обучения

- Построить автономного GitHub-агента issue-to-PR, работающего в песочнице.
- Планировать от issue, править код, запускать тесты и открывать pull request.
- Добавить гейты верификации и проход ревьюера перед отправкой.

## Задача

Async cloud coding agent — отдельная product category по сравнению с interactive coding agents (capstone 01). UX — это GitHub label. Ты ставишь label `@agent fix this` на issue, worker поднимается в cloud sandbox, клонирует repo, запускает tests, редактирует files, проверяет результат и открывает PR с rationale agent в body. Никакого interactive loop, никакого terminal. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules и Factory Droids сходятся к этому.

Engineering challenges конкретны: environment reproduction (agent должен собрать repo с нуля без cached dev image), flaky tests (их нужно перезапускать или изолировать), credential scoping (GitHub App с минимальными fine-grained permissions), budget enforcement per repo per day и no-force-push policy. Capstone измеряет pass rate, cost и safety vs hosted alternatives.

## Концепция

Trigger — GitHub webhook (issue label или PR comment). Dispatcher ставит work в queue для ECS Fargate или Lambda. Worker забирает repo в Daytona или E2B sandbox с generic Dockerfile, выведенным из repo (language, framework). Agent запускает mini-swe-agent или SWE-agent v2 loop с Claude Opus 4.7 или GPT-5.4-Codex. Он итерирует: read code, propose fix, apply patch, run tests.

Verification — gating step. Full CI должен пройти в sandbox до открытия PR. Coverage delta вычисляется; если она отрицательна сверх threshold, PR открывается, но получает label `needs-review`. Agent публикует rationale как PR description плюс thread `@agent`, куда reviewer может писать follow-ups.

Safety ограничивается через две разные GitHub surfaces: App предоставляет short-lived installation token с `workflows: read` и узкими repo contents/PR scopes; branch protection (не app permissions) обеспечивает "no direct writes to `main`" и "no force-push" — app никогда не добавляется в bypass list. Path-scoped read-only access к `.github/workflows` не является настоящим GitHub App primitive, поэтому worker должен enforce allow-list на file edits.

## Архитектура

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Стек

- Trigger: GitHub App с fine-grained token; webhook receiver через Lambda или Fly.io
- Worker: ECS Fargate task (или GitHub Actions self-hosted runner)
- Sandbox: Daytona devcontainer или E2B sandbox per task
- Agent loop: mini-swe-agent baseline или SWE-agent v2 over Claude Opus 4.7 / GPT-5.4-Codex
- Retrieval: tree-sitter repo-map + ripgrep
- Verification: full CI in-sandbox + coverage delta gate
- Observability: Langfuse с per-PR trace archive, ссылка на который есть в PR body
- Budget: per-repo daily dollar ceiling; max PRs per repo per day

## Сборка

1. **GitHub App.** Fine-grained installation token: issues read+write, pull_requests write, contents read+write, workflows read. Branch protection (единственная surface, которая может это сделать) обеспечивает "no direct push to `main`" и "no force-push"; app не находится в bypass list. Worker обеспечивает "no writes under `.github/workflows`" как allow-list check на proposed diff, потому что GitHub App permissions не являются path-scoped.

2. **Webhook receiver.** Lambda function принимает issue label / PR comment webhooks. Фильтрует по label `@agent fix this`. Ставит в SQS.

3. **Dispatcher.** Забирает tasks из SQS. Enforces per-repo per-day budget. Поднимает ECS Fargate task с repo URL, issue body и свежим Daytona sandbox.

4. **Environment inference.** Определи language (Python, Node, Go, Rust) и package manager (uv, pnpm, go mod, cargo). Сгенерируй Dockerfile на лету, если он отсутствует.

5. **Agent loop.** mini-swe-agent или SWE-agent v2 с Claude Opus 4.7. Tools: ripgrep, tree-sitter repo-map, read_file, edit_file, run_tests, git. Hard limits: $20 cost, 30 min wall-clock, 30 agent turns.

6. **Verification.** После завершения loop запусти full test suite in-sandbox. Посчитай coverage delta через jacoco / coverage.py. Если CI red: остановись, не открывай PR. Если coverage падает больше чем на 2%: открой PR с label `needs-review`.

7. **PR posting.** Push agent branch. Открой PR через GitHub API с: title, rationale, diff summary, trace URL, cost, turns.

8. **Credential hygiene.** Worker работает с short-lived GitHub App installation token. Logs scrubbed for secrets before archival.

9. **Eval.** 30 seeded internal issues разной сложности. Измерь pass rate, PR quality (diff size, style, coverage), cost, latency. Сравни с Cursor Background Agents и AWS Remote SWE Agents на тех же issues.

## Использование

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Что сдать

`outputs/skill-issue-to-pr.md` — deliverable. GitHub App + async cloud worker, который превращает labeled issues в review-ready PRs с bounded cost и scoped credentials.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Pass rate на 30 issues | End-to-end success (CI green + coverage OK) |
| 20 | PR quality | Diff size, coverage delta, соответствие style |
| 20 | Cost and latency per resolved issue | $ и wall-clock per PR |
| 20 | Safety | Scoped token, per-repo budget, no force-push, credential hygiene |
| 15 | Operator UX | Rationale comments, возможность retry, @-mention follow-up |
| **100** | | |

## Упражнения

1. Добавь режим "fix flaky test": label `@agent stabilize-flake TestX` запускает test 50 раз in-sandbox и предлагает minimal change, который стабилизирует его.

2. Сравни cost vs Cursor Background Agents на трех общих issues. Отчитай, какие tools где выигрывают.

3. Реализуй budget dashboard: per-repo per-day cost, per-user cost. Alert on anomaly.

4. Построй "dry-run" mode, который открывает draft PR без запуска CI, чтобы reviewers могли дешево изучить plan.

5. Добавь retention policy: PR branches старше 7 days без merge автоматически удаляются.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|------------------------|
| GitHub App | "Scoped bot identity" | App с fine-grained permissions + short-lived installation token |
| Async cloud agent | "Background agent" | Non-interactive worker, который работает в cloud sandbox, а не в terminal |
| Environment inference | "Синтез Dockerfile" | Определить language + package manager, сгенерировать Dockerfile при отсутствии |
| Verification | "CI-in-sandbox" | Запуск full test suite внутри worker перед открытием PR |
| Coverage delta | "Coverage preservation" | Изменение test coverage % от base до agent branch |
| Per-repo budget | "Дневной лимит" | Лимит в dollars и PR-count, enforced at the dispatcher |
| Rationale | "PR body explanation" | Summary agent о том, что изменилось и почему; обязательно в PR body |

## Дополнительное чтение

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — canonical reference для async cloud agent
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI reference
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — commercial alternative
- [OpenAI Codex (cloud)](https://openai.com/codex) — hosted competitor
- [Google Jules](https://jules.google) — hosted version от Google
- [Factory Droids](https://www.factory.ai) — alternate commercial reference
- [GitHub App documentation](https://docs.github.com/en/apps) — scoped bot identity
- [Daytona cloud sandboxes](https://daytona.io) — reference sandbox
