# Evaluation and Coordination Benchmarks

> Пять benchmarks 2025-2026 покрывают пространство multi-agent evaluation. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) оценивает topologies star/chain/tree/graph с milestone KPIs; **graph is best for research**, cognitive planning добавляет ~3% milestone achievement. **COMMA** оценивает multimodal asymmetric-information coordination; state-of-the-art models, включая GPT-4o, с трудом превосходят random baseline. **MedAgentBoard** (arXiv:2505.12371) покрывает четыре medical task categories и часто показывает, что multi-agent не доминирует над single-LLM. **AgentArch** (arXiv:2509.10769) бенчмаркит enterprise agent architectures, объединяющие tool-use + memory + orchestration. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) содержит 1865 problems across 41 repos, охватывающих business apps, B2B services и developer tools; frontier models набирают ~23% на Pro против 70%+ на Verified — reality check на contamination. Claude Opus 4.7 (апрель 2026) reported at **64.3%** on Pro with explicit agent-teams coordination (primary source Anthropic пока не опубликован — treat as preliminary); Verdent (agent scaffold) достигает **76.1% pass@1** on Verified ([Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **AAAI 2026 Bridge Program WMAC** (https://multiagents.org/2026/) — community focal point 2026 года. Этот урок строится на metrics MARBLE, запускает topology-vs-metric sweep и закрепляет правило "just passing SWE-bench Verified is not evidence of generalization".

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**Время:** ~75 минут

## Цели обучения

- Сопоставлять мультиагентные бенчмарки 2025-2026 (MultiAgentBench/MARBLE, COMMA и прочие) с тем, что каждый измеряет.
- Читать оценки координации по топологиям.
- Выбирать бенчмарк под конкретное мультиагентное утверждение.

## Проблема

Когда paper утверждает "our multi-agent system is better", вопрос: лучше чего, на чем и как измерено? Эпоха 2023-2024 в multi-agent evaluation была хаосом — каждый выбирал свои metrics, свои baselines и свои task sets. Benchmarks 2025-2026 внесли структуру.

Без shared benchmarks вы не можете meaningful compare две multi-agent systems. Хуже того, без hold-out benchmarks frontier models могут contaminate. SWE-bench Verified стал частично contaminated в training corpora к середине 2025; frontier scores inflated; Pro был спроектирован как uncontaminated reality check.

Этот урок перечисляет пять канонических benchmarks 2026 года, называет, что каждый измеряет, и учит skeptically читать benchmark claims.

## Концепция

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935. Оценивает четыре coordination topologies (star, chain, tree, graph) на research, coding и planning tasks. Milestone-based KPIs отслеживают partial progress, а не только final success.

Измеренные results:

- **Graph** topology best for research scenarios; поддерживает any-to-any critique.
- **Chain** best for stepwise-refinement coding.
- **Star** best for fast-factual consolidation.
- **Coordination tax** появляется после ~4 agents on graph.
- **Cognitive planning** добавляет ~3% milestone achievement across topologies.

Используйте, когда: вы хотите сравнить coordination topologies apples-to-apples. MARBLE repo (https://github.com/ulab-uiuc/MARBLE) предоставляет evaluator.

### COMMA — multimodal asymmetric information

Покрывает tasks, где agents имеют разные observation modalities и должны coordinate без full information sharing. Reported result неприятен: frontier models, включая GPT-4o, с трудом превосходят **random baseline** на agent-agent collaboration в COMMA. Сигнал в том, что multi-agent modalities under-trained и under-evaluated — LLMs неплохо справляются с single-modality cooperation; multi-modality coordination collapses.

Используйте, когда: ваша система имеет multimodal или asymmetric-information coordination. Null result из COMMA — предупреждение измерять до claims.

### MedAgentBoard — domain stress test

arXiv:2505.12371. Четыре medical task categories: diagnosis, treatment planning, report generation, patient communication. Сравнивает multi-agent vs single-LLM vs conventional rule-based systems.

Finding: multi-agent НЕ доминирует над single-LLM в большинстве categories. Multi-agent advantage узок — task decomposition помогает, когда subtasks clearly separable (diagnosis + treatment); вредит, когда coordination overhead превышает specialization gain (report generation).

Используйте, когда: у вашего domain есть clear-cut single-LLM baselines. Если lesson MedAgentBoard generalizes, многие proposed multi-agent systems over-engineered.

### AgentArch — enterprise architectures

arXiv:2509.10769. Enterprise settings с tool use, memory и orchestration layered together. Benchmark изолирует contribution каждого layer: насколько помогает adding tools? Adding memory? Adding multi-agent orchestration?

Используйте, когда: вы проектируете enterprise agent stack и нужно обосновать каждый layer. AgentArch помогает избегать покупки features, whose value cannot be measured.

### SWE-bench Pro — reality check

arXiv:2509.16941. 1865 problems across 41 repositories, охватывающих business apps, B2B services и developer tools. Спроектирован как **uncontaminated** with later training cutoffs. Frontier models набирают ~23% на Pro против 70%+ на Verified. Gap — contamination signal.

April 2026 scores:
- Claude Opus 4.7 on Pro: **64.3%** (reported with explicit agent-teams coordination; primary source Anthropic пока не опубликован — treat as preliminary).
- Verdent (agent scaffold) on Verified: **76.1% pass@1** ([technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Frontier raw scores on Pro without agent scaffolding: ~23-35% ([SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)).

Вывод: "we beat SWE-bench Verified" больше не является evidence of capability. Pro — текущий gating test. Agent-team scaffolding дает measurable gains on Pro (~30-40 point delta), что является одним из strongest empirical arguments for multi-agent coordination in 2026.

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination (https://multiagents.org/2026/). Community focal point 2026 года для multi-agent AI research. Accepted papers и workshop proceedings — каноническая venue для evaluation новых methods; для production decisions отдавайте предпочтение WMAC-accepted claims перед arXiv preprints.

### Читайте benchmark claims skeptically — checklist 2026 года

Когда кто-то claims a multi-agent result:

1. **Which benchmark, which split?** SWE-bench Verified vs Pro matters a lot. Число, reported on the wrong split, worthless.
2. **Contamination check.** Был ли benchmark released after the model's training cutoff? Если нет, treat with caution.
3. **Baseline comparison.** Vs single-LLM baseline, vs random, vs prior multi-agent work. Не "vs untuned version of the same system."
4. **Statistical significance.** N trials, p-value, confidence interval. Frontier models high-variance; single runs mislead.
5. **Task diversity.** One task or many? Generalization matters for production.
6. **Cost disclosure.** Tokens per task, wall-clock. 90% solution at 20x cost — business decision, not a capability claim.

### Что benchmarks плохо измеряют

- **Long-horizon coordination.** Days of wall-clock interaction. All current benchmarks run short.
- **Adversarial resilience.** Что происходит, когда один agent malicious или compromised?
- **Drift under deployment.** Benchmarks static; production distributions shift.
- **Cost-normalized performance.** Большинство benchmarks report raw accuracy, not accuracy-per-dollar.

Построение собственного internal benchmark для оси, которая вам реально важна, часто правильный ход.

## Соберите

`code/main.py` — non-interactive walk-through:

- Симулирует 3 multi-agent systems on a toy task.
- Вычисляет MARBLE-style milestone metrics для каждой.
- Запускает contamination check by withholding tasks from a "training" set.
- Explicitly compares to a random baseline.
- Печатает benchmark-claims scorecard.

Запуск:

```bash
python3 code/main.py
```

Ожидаемый output: system scorecard с raw accuracy, milestone achievement, cost-per-task, vs-random baseline delta и contamination-check note.

## Используйте

`outputs/skill-benchmark-reader.md` читает любой multi-agent benchmark claim и применяет scrutiny checklist. Вывод: оценку и оговорки.

## Запустите в production

Дисциплина production evaluation:

- **Build an internal benchmark** that reflects your actual production distribution. Public benchmarks inform but do not substitute.
- **Include a random baseline** in every comparison. Если вы не можете beat random by a large margin на coordination task, task may be ill-posed.
- **Report cost alongside accuracy.** Token cost и wall-clock. Ops teams need both.
- **Rebuild the benchmark quarterly.** Production distribution shifts; stale benchmarks mislead.
- **Avoid published-benchmark overfitting.** Если ваша команда оптимизируется specifically for SWE-bench Pro numbers, вы regress on production.

## Упражнения

1. Запустите `code/main.py`. Определите, какая из трех simulated systems имеет лучший cost-per-milestone. Совпадает ли это с system с highest raw-accuracy?
2. Прочитайте MultiAgentBench (arXiv:2503.01935). Для своего task domain решите, какую из четырех topologies MARBLE would recommend. Обоснуйте по results paper.
3. Прочитайте SWE-bench Pro paper. Что конкретно делает его contamination-resistant? Можно ли применить тот же technique к другим benchmarks, которые вам важны?
4. Прочитайте finding COMMA on multimodal coordination. Спроектируйте simple multimodal coordination task, который можно добавить к internal benchmark. Что считалось бы useful signal?
5. Примените benchmark-claims checklist к headline result одного recent multi-agent paper. Какую grade вы бы дали claim?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| MARBLE | "MultiAgentBench" | ACL 2025; star/chain/tree/graph topologies with milestone KPIs. |
| COMMA | "Multimodal benchmark" | Multimodal asymmetric-info coordination; frontier models struggle vs random. |
| MedAgentBoard | "Domain stress test" | Four medical categories; often finds multi-agent does not dominate single-LLM. |
| AgentArch | "Enterprise benchmark" | Tools + memory + orchestration layered. |
| SWE-bench Pro | "Contamination-resistant" | 1865 problems, 41 repos; ~23% vs 70%+ on Verified (the contamination signal). |
| Milestone achievement | "Partial credit" | Benchmarks that reward progress, not only final success. |
| Contamination | "Benchmark leaked into training" | Post-release, benchmarks drift into training corpora; scores inflate. |
| WMAC | "AAAI 2026 Bridge Program" | Workshop on Multi-Agent Coordination; community focal point. |

## Дополнительное чтение

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark with milestone KPIs
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — reference implementation
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — domain stress test; multi-agent often does not dominate
- [AgentArch](https://arxiv.org/abs/2509.10769) — enterprise agent architectures
- [SWE-bench leaderboards](https://www.swebench.com/) — Verified and Pro scores for frontier models
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — the 2026 community focal point
