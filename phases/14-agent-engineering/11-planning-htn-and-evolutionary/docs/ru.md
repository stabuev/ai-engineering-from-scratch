# Планирование с HTN и эволюционным поиском

> Symbolic planning покрывает случаи, где план доказуемо корректен. Evolutionary code search покрывает случаи, где fitness function проверяема машиной. ChatHTN (2025) и AlphaEvolve (2025) показывают, что дает каждый подход в паре с LLM.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 02 (ReWOO и Plan-and-Execute)
**Время:** ~75 минут

## Цели обучения

- Объяснить Hierarchical Task Networks: tasks, methods, operators, preconditions, effects.
- Описать гибридный цикл ChatHTN — symbolic search с LLM fallback decomposition.
- Объяснить evolutionary loop AlphaEvolve и почему он работает только с programmatic evaluator.
- Реализовать игрушечный HTN planner плюс игрушечный evolutionary search на stdlib.

## Проблема

ReWOO (Lesson 02), Plan-and-Execute и ReAct покрывают большую часть agent planning. Два случая они покрывают плохо:

1. **Планы с доказуемой корректностью.** Scheduling, flight pathing, compliance workflows — план должен быть sound by construction. Fluent LLM plan, который иногда hallucinate шаг, неприемлем.
2. **Оптимизации с machine-checkable fitness function.** Matrix multiplication, scheduling heuristics, compiler passes — цель не "корректный план", а "лучший план."

HTN planning и AlphaEvolve решают две разные проблемы. Оба используют LLM как усилители, а не замену.

## Концепция

### Hierarchical Task Networks

HTN состоит из:

- **Tasks** — compound (нужно декомпозировать) и primitive (можно выполнить напрямую).
- **Methods** — способы декомпозировать compound task в subtasks, с preconditions.
- **Operators** — primitive actions с preconditions и effects.
- **State** — множество фактов.

Planning: given goal task и initial state, найти decomposition в primitive operators, чьи preconditions последовательно выполняются.

HTN старше LLM и до сих пор является reference для provably-correct plans.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814) чередует symbolic HTN с LLM queries:

1. Попробовать декомпозировать текущую compound task существующими methods.
2. Если ни один method не применим, спросить LLM: "how would you decompose `task` in state `s`?"
3. Перевести ответ LLM в candidate subtasks.
4. Валидировать по operator schema; отклонить invalid decompositions.
5. Рекурсировать.

Центральное утверждение статьи: каждый произведенный план доказуемо sound, потому что LLM suggestions входят только как candidate decompositions, а не как прямые правки плана. Symbolic layer отвечает за корректность; LLM расширяет method library.

Online method learning (OpenReview `gwYEDY9j2x`, 2025 follow-up) добавляет learner, который обобщает LLM-produced decompositions через regression — сокращая частоту LLM queries до 75%.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, June 2025) — другой зверь: evolutionary code search, оркестрированный ensemble Gemini 2.0 Flash/Pro.

Цикл:

1. Начать с seed program + programmatic evaluator (возвращает fitness score).
2. Ensemble LLMs предлагает mutations.
3. Запустить mutations через evaluator.
4. Сохранить лучшие; мутировать снова.

Опубликованные победы:

- Первое улучшение Strassen для 4x4 complex matrix multiplication за 56 лет (48 scalar multiplications).
- 0.7% recovered Google compute через Borg scheduling heuristic.
- 32% FlashAttention speedup на frontier workload.

Жесткое ограничение: fitness function должна быть machine-checkable. Evolutionary search по prose answers не сходится.

### Что когда использовать

| Класс проблемы | Используйте | Почему |
|---------------|-----|-----|
| Scheduling с жесткими constraints | HTN + ChatHTN | Provable soundness |
| Compiler optimization | AlphaEvolve | Machine-checkable fitness |
| Multi-step task execution | ReAct / ReWOO | LLM in the loop, без formal guarantees |
| Улучшение кода с tests | AlphaEvolve | Tests are the evaluator |
| Policy-bound automation | HTN | Preconditions encode policy |

### Где этот паттерн ломается

- **HTN без operators.** Без schemas precondition/effect утверждение о soundness разваливается. ChatHTN "LLM suggests decomposition" требует schema, которая отклоняет invalid moves.
- **AlphaEvolve без реального evaluator.** "Спросить LLM, лучше ли код" — это не fitness function. Evaluator должен быть deterministic и fast.
- **Over-engineering.** Большинству agent tasks не нужно ни то ни другое. Сначала беритесь за ReAct или ReWOO.

## Соберите это

`code/main.py` реализует две игрушки:

- Stdlib HTN planner с operators, methods, preconditions, effects и `LLMFallback`, который включается, когда ни один method не подходит compound task. "LLM" — скриптовый decomposer, чтобы planner работал offline.
- Stdlib evolutionary search по arithmetic programs: выращивает expressions, чей output минимизирует `|f(x) - target|` на test set. Evaluator deterministic.

Запустите:

```
python3 code/main.py
```

Трасса показывает, как HTN planner декомпозирует compound task (с mid-plan LLM fallback), и как evolutionary loop сходится к target expression.

## Используйте это

- **HTN planners** — `pyhop`, `SHOP3` или собственная реализация для domain-specific policy enforcement.
- **ChatHTN** — research code; паттерн (symbolic + LLM fallback) чисто переносится в любой HTN planner.
- **AlphaEvolve** — статья DeepMind; паттерн (ensemble + evaluator) воспроизводим. OpenEvolve и похожие open-source forks появляются.
- **Agent frameworks** — пока ни один не поставляет first-class HTN или AlphaEvolve. Стройте это как subagent или background worker.

## Доведите до продакшена

`outputs/skill-hybrid-planner.md` генерирует scaffold hybrid planner (HTN или evolutionary) с явно ограниченной ролью LLM.

## Упражнения

1. Расширьте HTN planner backtracking: когда postcondition operator fails at runtime, откатиться и попробовать следующий method.
2. Добавьте LLM-method cache в ChatHTN: когда LLM декомпозирует task `T` в state pattern `P`, сохраните результат. На следующем вызове сначала заново проверьте method library.
3. Замените evaluator evolutionary search на реальный test suite. Эволюционируйте sort function, которая проходит 20 test cases; сообщите generations to convergence.
4. Прочитайте design notes evaluator в AlphaEvolve. Спроектируйте evaluator для важной вам области (SQL query optimization, test-suite minimization, deployment YAML).
5. Скомбинируйте: используйте HTN для декомпозиции compound task на subtasks, затем используйте evolutionary search на primitive operator каждой subtask. Где это хорошо работает, а где over-engineer?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| HTN | "Hierarchical planner" | Декомпозиция tasks с operators, preconditions, effects |
| Method | "Decomposition rule" | Способ разбить compound task на subtasks |
| Operator | "Primitive action" | Конкретный шаг с precondition и effect |
| ChatHTN | "LLM + HTN" | Symbolic planner спрашивает LLM, когда ни один method не подходит |
| AlphaEvolve | "Evolutionary code search" | Ensemble LLMs мутируют код; deterministic evaluator отбирает |
| Fitness function | "Evaluator" | Детерминированная, machine-checkable оценка outputs |
| Online method learning | "Cached LLM decomposition" | Store + generalize LLM plans, чтобы снизить query cost |

## Дополнительное чтение

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — symbolic + LLM hybrid planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — evolutionary code search с LLM mutations
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — когда брать planner, а когда simple loop
