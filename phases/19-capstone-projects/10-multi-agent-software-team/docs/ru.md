# Capstone 10 — Мультиагентная команда разработки ПО

> SWE-AF factory architecture, role-based prompting в MetaGPT, typed actor graph в AutoGen 0.4, Devin от Cognition и Droids от Factory сошлись к одной форме 2026 года: architect планирует, N coders работают параллельно в worktrees, reviewer ставит gate, tester проверяет. Parallel worktrees превращают wall-clock во throughput. Shared state и handoff protocols становятся поверхностью отказов. Capstone: построить team, оценить на SWE-bench Pro и отчитаться, какие handoffs ломаются и как часто.

**Тип:** Capstone
**Языки:** Python / TypeScript (agents), Shell (worktree scripts)
**Предварительные требования:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Отрабатываемые фазы:** P11 · P13 · P14 · P15 · P16 · P17
**Время:** 40 часов

## Цели обучения

- Построить многоагентную команду разработки с ролевой специализацией и типизированным графом акторов/сообщений.
- Координировать планирование, кодинг и ревью между агентами.
- Добавить общее состояние и гейт верификации перед слиянием.

## Проблема

Single-agent coding harnesses упираются в потолок на больших задачах. Не потому, что отдельный agent слаб, а потому что 200k-token context не может вместить architecture plan плюс четыре параллельных среза codebase плюс reviewer commentary плюс test output. Multi-agent factories делят проблему: architect владеет plan, coders владеют implementation в parallel worktrees, reviewer ставит gate, tester проверяет. "Factory" architecture SWE-AF, roles MetaGPT, typed actor graph AutoGen - все три описания задают одну и ту же форму.

Поверхность отказов - это handoff. Architect планирует то, что coders не могут реализовать. Coders создают конфликтующие diffs. Reviewer одобряет hallucinated fix. Tester соревнуется с coder, который еще пишет. Вы построите одну из таких команд, запустите ее на 50 issues из SWE-bench Pro, отследите каждый handoff и опубликуете post-mortem.

## Концепция

Roles - это типизированные agents. **Architect** (Claude Opus 4.7) читает issue, пишет plan и разбивает его на subtasks с явными interfaces. **Coders** (Claude Sonnet 4.7, N parallel instances, каждый в `git worktree` + Daytona sandbox) независимо реализуют subtasks. **Reviewer** (GPT-5.4) читает merged diff и либо одобряет, либо запрашивает конкретные changes. **Tester** (Gemini 2.5 Pro) запускает test suite в изоляции и сообщает pass/fail с artifacts.

Коммуникация идет через shared task board (file-backed или Redis). Каждая role потребляет tasks, которые ей разрешено обрабатывать. Handoffs - это A2A-protocol-typed messages. Coordination concerns: merge-conflict resolution (coordinator role или automatic three-way merge), shared-state synchronization (plan замораживается, когда coders стартуют; replans являются отдельными events) и reviewer gatekeeping (reviewer не может approve собственные changes или changes, которые сам предложил).

Token amplification - скрытая стоимость. Каждая role boundary добавляет summary prompts и handoff context. 40-turn single-agent run превращается в 160 total turns через четыре roles. Rubric отдельно взвешивает token efficiency vs single-agent baseline, потому что вопрос не "работает ли multi-agent", а "выигрывает ли он per dollar."

## Архитектура

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Стек

- Orchestration: LangGraph с shared state + per-agent sub-graphs
- Messaging: A2A protocol (Google 2025) для типизированных inter-agent messages
- Models: Opus 4.7 (architect), Sonnet 4.7 (coders), GPT-5.4 (reviewer), Gemini 2.5 Pro (tester)
- Worktree isolation: `git worktree add` per coder + Daytona sandbox
- Merge coordinator: custom three-way merge + LLM-mediated conflict resolution
- Eval: SWE-bench Pro (50 issues), SWE-AF scenarios, HumanEval++ для unit tests
- Observability: Langfuse с role-tagged spans, per-agent token accounting
- Deployment: K8s, где каждая role развернута как отдельный Deployment + HPA по backlog

## Соберите

1. **Task board.** File-backed JSONL с typed messages: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Agents подписываются на tags.

2. **Architect.** Читает GitHub issue, запускает Opus 4.7 с plan template, требующим явных subtask interfaces (files touched, public functions, test impact). Эмитит один `plan_request` с DAG subtasks.

3. **Coders.** N parallel workers, каждый забирает один subtask с board. Каждый создает свежую `git worktree add` branch плюс Daytona sandbox. Реализует subtask. Эмитит `diff_ready` с patch + test deltas.

4. **Merge coordinator.** Когда все coders завершили работу, выполняет three-way merge N branches в staging branch. LLM-mediated conflict resolution используется только при file-level overlap.

5. **Reviewer.** GPT-5.4 читает merged diff. Не может approve diffs, которые сам authored. Эмитит `approved` (no-op) или `review_feedback` с specific change requests, routed back к соответствующему coder.

6. **Tester.** Gemini 2.5 Pro запускает test suite в clean sandbox. Сохраняет artifacts. Эмитит `test_passed` или `test_failed` со stacktraces. Failed tests возвращаются к coder, владеющему failing subtask.

7. **Handoff accounting.** Каждое message, пересекающее role boundary, получает span в Langfuse с payload size и использованной model. Вычислите per-subtask token amplification (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens).

8. **Eval.** Запустите на 50 issues из SWE-bench Pro. Сравните pass@1 и $-per-solved-issue с single-agent baseline (один Sonnet 4.7 в single worktree).

9. **Post-mortem.** Для каждого failed issue определите handoff, который сломался (слишком расплывчатый plan, merge conflict, reviewer false-approve, tester flake). Создайте гистограмму handoff-failure.

## Используйте

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Сдайте

`outputs/skill-multi-agent-team.md` - deliverable. Для заданного issue URL и уровня parallelism команда создает merge-ready PR с per-role token accounting.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | Согласованное подмножество из 50 issues, pass@1 |
| 20 | Parallel speedup | Wall-clock vs single-agent baseline |
| 20 | Review quality | False-approval rate on injected-bug probe |
| 20 | Token efficiency | Total tokens на solved issue относительно single-agent |
| 15 | Coordination engineering | Merge-conflict resolution, гистограмму handoff-failure |
| **100** | | |

## Упражнения

1. Вставьте очевидный bug в diff mid-run (лишний `return None` перед main body). Измерьте reviewer false-approve rate. Настраивайте reviewer prompt, пока false-approval не станет ниже 5%.

2. Сократите до two coders (architect + coder + reviewer + tester, coder выполняет two subtasks sequentially). Сравните wall-clock и pass rate.

3. Замените merge coordinator на single-writer constraint (subtasks touch disjoint file sets). Измерьте planning burden на architect.

4. Замените reviewer с GPT-5.4 на Claude Opus 4.7. Измерьте false-approval rate и token cost delta.

5. Добавьте fifth role: documenter (Haiku 4.5). После review он создает changelog entry. Измерьте, оправдывает ли documentation quality дополнительные token spend.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Parallel worktree | "Isolated branch" | `git worktree add`, создающий свежий working tree для каждого coder |
| Task board | "Shared message bus" | File или Redis store typed messages, на которые подписываются agents |
| Handoff | "Role boundary" | Любое message, пересекающее context одной role и попадающее в context другой |
| Token amplification | "Multi-agent overhead" | Total tokens по roles / single-agent tokens для той же task |
| A2A protocol | "Agent-to-agent" | Спецификация Google 2025 для typed inter-agent messages |
| Merge coordinator | "Integrator" | Component, который выполняет three-way merge и mediates conflicts |
| False approval | "Reviewer hallucination" | Reviewer одобряет diff с известными bugs |

## Дополнительное чтение

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — reference multi-agent factory 2026 года
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — role-based multi-agent framework
- [AutoGen v0.4](https://github.com/microsoft/autogen) — typed actor framework от Microsoft
- [Cognition AI (Devin)](https://cognition.ai) — reference product
- [Factory Droids](https://www.factory.ai) — альтернативный reference product
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — спецификация inter-agent messaging
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — isolation substrate
- [SWE-bench Pro](https://www.swebench.com) — цель оценки
