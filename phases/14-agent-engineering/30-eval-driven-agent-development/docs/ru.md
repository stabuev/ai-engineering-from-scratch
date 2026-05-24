# Разработка агентов через evals

> Рекомендация Anthropic: "start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when needed." Evaluation — не последний шаг. Это внешний цикл, который управляет каждым другим выбором в Фазе 14.

**Тип:** Изучение + сборка
**Языки:** Python (stdlib)
**Предварительные требования:** вся Фаза 14.
**Время:** ~60 минут

## Цели обучения

- Назвать три слоя evaluation — статические бенчмарки, кастомные offline evals и online evals в production — и назначение каждого.
- Объяснить плотный цикл evaluator-optimizer.
- Описать best practice 2026: evals живут рядом с кодом, запускаются в CI и блокируют PR.
- Связать каждый урок Фазы 14 с eval case, который он порождает.

## Проблема

Агенты проходят демо. Они ломаются в production способами, которые демо не могут предсказать. Бенчмарки отвечают на вопрос "is this model broadly capable?", а не "is this agent shipping the right patches for my product?" Ответ: evaluation на трех слоях, непрерывно запускаемая, где каждый guardrail и learned rule сопоставлены с eval case.

## Концепция

### Три слоя evaluation

1. **Статические бенчмарки** — SWE-bench Verified для кода (урок 19), WebArena/OSWorld для браузера / desktop (урок 20), GAIA для generalist-задач (урок 19), BFCL V4 для tool use (урок 06). Используйте их для сравнения моделей и regression gating. Contamination реален: SWE-bench+ нашел 32.67% solution leakage. Всегда сообщайте Verified / +-audited scores.

2. **Кастомные offline evals** — форма вашего продукта:
   - LLM-as-judge (Langfuse, Phoenix, Opik — Lesson 24).
   - Execution-based (запустить patch, проверить tests).
   - Trajectory-based (сравнить action sequences с gold; OSWorld-Human показывает, что top agents делают 1.4-2.7x шагов относительно gold).

3. **Online evals** — production:
   - Session replays (Langfuse).
   - Guardrail-triggered alerts (Lesson 16, 21).
   - Per-step cost / latency tracking (Lesson 23 OTel spans).

### Evaluator-optimizer (Anthropic)

Плотный цикл:

1. Proposer генерирует output.
2. Evaluator оценивает.
3. Refine, пока evaluator не пройдет.

Это Self-Refine (урок 05), обобщенный. Любой важный agent flow можно обернуть в evaluator-optimizer для надежности.

### Лучшие практики 2026 года

- Evals живут рядом с кодом.
- Запускаются в CI на каждом PR.
- Merge блокируется по eval scores (например, "no regression > 5% vs main").
- Каждый guardrail сопоставлен с eval case.
- Каждое learned rule (Reflexion, pro-workflow learn-rule) сопоставлено с failure case.

### Как Phase 14 связывается воедино

Каждый урок Фазы 14 порождает eval cases:

| Урок | Eval case, который он порождает |
|--------|------------------------|
| 01 Agent Loop | Budget-exhausted, infinite-loop guard |
| 02 ReWOO | Planner корректно replans при tool failure |
| 03 Reflexion | Learned reflections применяются при retry |
| 05 Self-Refine/CRITIC | Judge пропускает refined output |
| 06 Tool Use | Argument coercion работает; unknown tools отклоняются |
| 07-10 Memory | Retrieval citations совпадают с sources; stale facts invalidate |
| 12 Workflow Patterns | Каждый pattern производит правильный output |
| 13 LangGraph | Resume точно воспроизводит state |
| 14 AutoGen Actors | DLQ ловит crashed handlers |
| 16 OpenAI Agents SDK | Guardrail срабатывает на правильных inputs |
| 17 Claude Agent SDK | Subagent results возвращаются к orchestrator |
| 19-20 Benchmarks | SWE-bench Verified score, WebArena success rate, OSWorld efficiency |
| 21 Computer Use | Per-step safety ловит injected DOM |
| 23 OTel | Spans emit required attributes |
| 26 Failure Modes | Detectors tag known failures |
| 27 Prompt Injection | PVE refuses poisoned retrievals |
| 28 Orchestration | Supervisor routes to the right specialist |
| 29 Runtime Shapes | DLQ handles N% failure |

Если в вашем eval suite есть cases для каждого, Фаза 14 покрыта.

### Где eval-driven development ломается

- **Нет baseline.** Evals без last-known-good нечитаемы. Храните baselines.
- **LLM-judge без grounding.** Judges тоже hallucinate. CRITIC pattern (урок 05) — judge grounds on external tools.
- **Over-fitting к evals.** Оптимизация под eval расходится с production usefulness. Ротируйте cases.
- **Flaky evals.** Non-deterministic cases вызывают false alarms. Фиксируйте seeds, snapshot state.

## Практика

`code/main.py` — eval harness на stdlib:

- Реестр cases с категориями (benchmark, custom, online).
- Scripted agent under test.
- Цикл evaluator-optimizer: propose, judge, refine до pass или max rounds.
- CI gate: агрегированный pass rate + regression against baseline.

Запустите:

```
python3 code/main.py
```

Output: pass/fail по каждому case, regression flag, verdict CI gate.

## Как использовать

- Пишите eval cases в том же repo, что и код агента.
- Запускайте их на каждом PR через CI.
- Роняйте build при regression.
- Отслеживайте pass rate во времени.
- Связывайте каждый production failure с новым case.

## Что подготовить

`outputs/skill-eval-suite.md` строит трехслойный eval suite для agent product с CI gates и regression tracking.

## Упражнения

1. Возьмите один из ваших production failures. Напишите eval case, который его воспроизводит. Проходит ли ваш agent его сейчас?
2. Постройте LLM-judge rubric для вашего domain с тремя dimensions (factual, tone, scope). Оцените 50 sessions.
3. Подключите eval suite к CI. Роняйте build при regression >=5%.
4. Добавьте метрику trajectory efficiency: сколько steps сделал agent по сравнению с gold trajectory?
5. Сопоставьте каждый урок Фазы 14 с eval case в вашем suite. Есть пропуски? Это gap, который нужно закрыть.

## Ключевые термины

| Термин | Как говорят люди | Что это на самом деле значит |
|------|----------------|------------------------|
| Static benchmark | "Готовая eval" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | "Доменная eval" | LLM-as-judge / exec / trajectory под форму вашего продукта |
| Online eval | "Продакшен-eval" | Session replay, guardrail alerts, отслеживание cost/latency |
| Evaluator-optimizer | "Propose-judge-refine" | Итерации до тех пор, пока judge не пройдет |
| CI gate | "Блокер merge" | Роняет build при eval regression |
| Baseline | "Last-known-good" | Reference score для обнаружения регрессии |
| Trajectory efficiency | "Шаги сверх gold" | Число шагов агента, деленное на минимум human expert |

## Дополнительное чтение

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "start simple, optimize with evals"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool-use benchmark
- [Langfuse docs](https://langfuse.com/) — evals + session replay на практике
