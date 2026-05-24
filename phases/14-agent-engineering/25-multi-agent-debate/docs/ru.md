# Многоагентные дебаты и совместная работа

> Du et al. (ICML 2024, "Society of Minds") запускают N экземпляров модели, которые независимо предлагают ответы, а затем итеративно критикуют друг друга в течение R раундов, чтобы сойтись. Это улучшает factuality, rule-following, reasoning. Sparse topology выигрывает у full mesh по token cost.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Время:** ~60 минут

## Цели обучения

- Объяснить debate protocol: N proposers, R rounds, сходимость к общему ответу.
- Описать, почему debate улучшает factuality, rule-following и reasoning.
- Объяснить sparse topology: не каждому debater нужно видеть всех остальных.
- Реализовать stdlib debate поверх scripted LLM с full-mesh и sparse variants; измерить token cost vs accuracy.

## Проблема

Self-Refine (Lesson 05) - это одна модель, критикующая саму себя, что несет риск groupthink. CRITIC (Lesson 05) заземляет critique во external tools, но они не всегда доступны. Debate вводит третий режим: multiple instances, cross-critique, convergence by disagreement.

## Концепция

### Society of Minds (Du et al., ICML 2024)

- N экземпляров модели независимо предлагают ответы на один и тот же вопрос.
- В течение R раундов каждая модель читает proposals других и критикует их.
- Модели обновляют свои ответы на основе critiques.
- После R раундов возвращается convergent answer.

В исходных экспериментах использовались N=3, R=2 из-за стоимости. Accuracy улучшается с увеличением числа agents и rounds на сложных задачах (MMLU, GSM8K, Chess Move Validity, biography generation).

Cross-model combinations превосходят single-model debates: ChatGPT + Bard вместе > каждый по отдельности.

### Sparse topology

"Improving Multi-Agent Debate with Sparse Communication Topology" (arXiv:2406.11776, 2024-2025) показала, что full-mesh debate не всегда оптимален. Sparse topologies (star, ring, hub-and-spoke) могут давать сопоставимую accuracy при меньшем token cost. Каждый debater видит только подмножество peers.

Следствия:

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals, каждый читает 4 peers = 60 critique ops.
- Star N=5, R=3 (one hub + 4 spokes) = 15 proposals, spokes читают только hub = 12 critique ops.

### Когда debate помогает

- **Factuality.** N независимых proposals, cross-check снижает hallucination.
- **Rule-following.** Chess move validity — одна модель пропускает правило, другие замечают.
- **Open-ended reasoning.** Несколько framings сходятся к правильному ответу.

### Когда debate вредит

- **Latency-sensitive UX.** N × R serial rounds - это задержка, которой у вас может не быть.
- **Cost-sensitive scale.** N × R tokens на вопрос.
- **Простые factual lookups.** Один lookup дешевле пяти debates.

### Практические реализации 2026 года

- **Anthropic orchestrator-workers** (Lesson 12) — один вариант debate с synthesis step.
- **LangGraph supervisor** (Lesson 13) — central router + specialist agents могут реализовать debate как node.
- **OpenAI Agents SDK** (Lesson 16) — agents handoff туда и обратно для iterative critique.
- **Multi-agent evals** — combine debate + evaluator-optimizer для eval signal.

### Где этот паттерн ломается

- **Convergence collapse.** Все agents сходятся на первом неверном ответе. Смягчайте обязательными раундами disagreement.
- **Hub failure.** В star topology плохой hub заражает всех. Ротируйте hub или используйте несколько hubs.
- **Prompt homogenization.** Все agents используют один и тот же prompt; они выдают одинаковые ответы. Используйте diverse prompts и/или models.

## Соберите это

`code/main.py` реализует stdlib debate:

- Класс `Debater` (scripted LLM с per-debater opinion drift).
- Runners `FullMeshDebate` и `SparseDebate`.
- Три вопроса: factual, rule-based и reasoning.
- Metrics: convergent answer, rounds to convergence, total critique ops.

Запустите:

```
python3 code/main.py
```

Output: per-protocol accuracy and cost; sparse совпадает с full mesh на 2/3 questions при меньшей стоимости.

## Используйте это

- **Anthropic orchestrator-workers** для простых debates с 2-3 workers.
- **LangGraph** для stateful multi-round debate с checkpointing.
- **Custom** для research или specialized correctness guarantees.

## Доведите до продакшена

`outputs/skill-debate.md` формирует каркас multi-agent debate с configurable topology, N, R и convergence rule.

## Упражнения

1. Реализуйте правило "forced disagreement": в round 1 каждый debater должен выдать distinct proposal. Измерьте влияние на convergence speed.
2. Добавьте confidence-weighted aggregation: debaters возвращают (answer, confidence); aggregator взвешивает по confidence. Помогает ли это?
3. Замените одного "agent" на другую scripted LLM с другими opinions. Улучшает ли heterogeneity accuracy?
4. Измерьте token cost для full mesh vs sparse на ваших 3 questions. Постройте график cost vs accuracy.
5. Прочитайте paper Society of Minds. Перенесите toy на N=5, R=3. Что ломается? Что становится лучше?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Debate | "Multi-agent critique" | N proposers, R rounds cross-critique, сходимость |
| Full mesh | "Все читают всех" | Каждый участник дебатов читает каждого другого участника в каждом раунде |
| Sparse topology | "Ограниченный обзор соседей" | Участники дебатов читают только подмножество соседей |
| Hub-and-spoke | "Звездная топология" | Один центральный участник, N-1 spoke-участников читают только hub |
| Convergence | "Agreement" | Debaters сходятся на общем ответе |
| Society of Minds | "Debate paper Du et al." | ICML 2024 multi-agent debate method |

## Дополнительное чтение

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — canonical multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology results
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers как вариант debate
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — single-model self-critique counterpart
