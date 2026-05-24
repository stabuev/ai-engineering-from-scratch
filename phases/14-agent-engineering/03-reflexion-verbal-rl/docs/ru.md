# Reflexion: вербальное обучение с подкреплением

> Gradient-based RL нужны тысячи trials и GPU cluster, чтобы исправить failure mode. Reflexion (Shinn et al., NeurIPS 2023) делает это на естественном языке: после каждой failed trial агент пишет reflection, сохраняет её в episodic memory и conditioning следующей trial делает на этой памяти. Это паттерн за sleep-time compute в Letta, learnings в `CLAUDE.md` у Claude Code и `learn-rule` в pro-workflow.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 02 (ReWOO)
**Время:** ~60 минут

## Цели обучения

- Назвать три компонента Reflexion (Actor, Evaluator, Self-Reflector) и роль episodic memory.
- Реализовать Reflexion loop на stdlib с binary evaluator, reflection buffer и fresh re-attempts.
- Выбирать между scalar, heuristic и self-evaluated feedback sources для конкретной задачи.
- Объяснить, почему verbal reinforcement ловит ошибки, для исправления которых gradient-based RL потребовались бы тысячи trials.

## Проблема

Агент проваливает задачу. В стандартном RL вы бы запустили ещё тысячи trials, посчитали gradients, обновили weights. Дорого, медленно, и у большинства production agents нет training budget на каждую ошибку.

Reflexion (Shinn et al., arXiv:2303.11366) задаёт другой вопрос: что если агент просто подумает, почему он ошибся, и попробует снова с этой мыслью в prompt? Без weight updates. Без gradient. Только естественный язык, сохранённый между trials.

Результат: на ALFWorld он превосходит ReAct и другие non-fine-tuned baselines. На HotpotQA улучшает ReAct. В code generation (HumanEval/MBPP) устанавливает state of the art на тот момент. И всё это без единого gradient step.

## Концепция

### Три компонента

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

Плюс одна структура данных:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

Одна trial запускает Actor. Evaluator оценивает её. Если score низкий, Self-Reflector создаёт reflection ("I picked the wrong tool because I misread the question as asking about X when it was asking about Y"). Reflection попадает в episodic memory. Следующая trial начинается заново, но видит reflection.

### Три типа evaluator

1. **Scalar** — внешний binary signal. ALFWorld succeeds or fails. HumanEval tests pass or fail. Самый простой, самый high-signal.
2. **Heuristic** — заранее заданные сигнатуры сбоев. "Если агент выдал одно и то же действие два раза подряд, пометь как застрявший." "Если траектория превышает 50 шагов, пометь как неэффективную."
3. **Self-evaluated** — LLM оценивает собственную trajectory. Нужен, когда ground truth недоступен. Более слабый signal; хорошо сочетается с tool-grounded verification (Lesson 05 — CRITIC).

Default 2026 года — смесь: scalar, когда доступен; self-eval, когда нет; heuristics как safety rails.

### Почему это обобщается

Reflexion — не столько новый алгоритм, сколько именованный паттерн. Почти каждый production "self-healing" agent запускает какой-то вариант:

- Sleep-time compute Letta (Lesson 08): отдельный agent рефлексирует над прошлыми conversations и пишет в memory blocks.
- Паттерн `CLAUDE.md` / "save memory" в Claude Code: reflections фиксируются как learnings и prepended к будущим sessions.
- Команда `/learn-rule` в pro-workflow: corrections фиксируются как explicit rules.
- Reflection nodes в LangGraph: node, который scores output и routes to refine if needed.

Все они следуют одной insight: natural language — достаточно богатая среда, чтобы переносить "what I learned from failure" между runs.

### Когда это работает и когда нет

Reflexion работает, когда:

- Есть четкий failure signal (test failure, tool error, wrong answer).
- Task class воспроизводим (тот же тип вопроса можно задать снова).
- У reflection есть пространство для улучшения trajectory (достаточный action budget).

Reflexion не помогает, когда:

- Agent уже succeeds on the first try.
- Failure внешний (network down, tool broken) — reflection о том, что "the network was down", не помогает будущим runs.
- Reflection превращается в superstition — хранение narrative об одноразовом flaky run.

Ловушка 2026 года: memory rot. Reflections накапливаются; некоторые устаревают или неверны; re-runs становятся медленнее по мере роста episodic buffer. Mitigation: periodic compaction (Lesson 06), TTL on reflections или отдельный sleep-time cleanup agent (Letta).

## Соберите это

`code/main.py` реализует Reflexion на игрушечной задаче: создать 3-element list, сумма которого равна target. Actor выдаёт candidate lists; Evaluator проверяет sum; Self-Reflector пишет строку о том, что пошло не так. Reflection попадает в episodic memory для следующей trial.

Компоненты:

- `Actor` — scripted policy, которая улучшается, когда видит reflections.
- `Evaluator.binary()` — pass/fail по target sum.
- `SelfReflector` — создаёт one-line diagnosis failure.
- `EpisodicMemory` — bounded list с TTL semantics.

Запустите:

```
python3 code/main.py
```

Trace показывает три trials. Trial 1 завершается неудачей, reflection сохраняется, trial 2 видит reflection и улучшается, но всё ещё не проходит, trial 3 успешен. Сравните с baseline run (без reflection) — он застревает на ответе trial 1.

## Используйте это

LangGraph поставляет reflection как node pattern. Команда `/memory` в Claude Code и `/learn-rule` в pro-workflow выносят episodic buffer в markdown file. Sleep-time compute Letta запускает Self-Reflector during downtime, чтобы primary agent оставался latency-bound. OpenAI Agents SDK не поставляет Reflexion напрямую; вы строите его через custom Guardrail, который rejects trajectories by score, и memory `Session`, которая survives across runs.

## Отгрузите это

`outputs/skill-reflexion-buffer.md` создаёт и поддерживает episodic buffer с reflection capture, TTL и deduplication. При заданном task class и failure он выдаёт reflection, которая действительно помогает следующей trial (а не generic "be more careful").

## Упражнения

1. Переключитесь с binary на scalar evaluator, который возвращает distance metric (насколько далеко от target). Сходится ли быстрее?
2. Добавьте TTL в 10 trials для reflections. Старые reflections после этого вредят или помогают?
3. Реализуйте heuristic evaluator: mark the trial as stuck, если одно и то же action repeats. Как это взаимодействует с Self-Reflector?
4. Запустите Reflexion с adversarial Actor, который ignores reflections. Какой минимум reflection prompt engineering заставит Actor заметить их?
5. Прочитайте Section 4 статьи Reflexion об AlfWorld. Концептуально воспроизведите 130% success-rate improvement: какой key delta относительно vanilla ReAct?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Reflexion | "Самокоррекция" | Shinn et al. 2023 — Actor, Evaluator, Self-Reflector плюс episodic memory |
| Verbal reinforcement | "Learning without gradients" | Natural-language reflection, prepended к prompt следующей trial |
| Episodic memory | "Per-task reflections" | Bounded buffer предыдущих reflections для одного task class |
| Scalar evaluator | "Бинарный success signal" | Pass/fail или numeric score из ground truth |
| Heuristic evaluator | "Pattern-based detector" | Заранее заданные сигнатуры сбоев (например, stuck-loop, too-many-steps) |
| Self-evaluator | "LLM-as-judge on own trace" | Lower-signal fallback без ground truth — сочетайте с tool-grounded verification |
| Memory rot | "Stale reflections" | Episodic buffer заполняется obsolete entries; fix через compaction/TTL |
| Sleep-time reflection | "Async self-reflection" | Запуск Self-Reflector off the hot path, чтобы primary agent оставался быстрым |

## Дополнительное чтение

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — каноническая статья
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — async reflection in production
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — управление episodic buffer как частью context
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reflection node pattern
