# Consensus and Byzantine Fault Tolerance for Agents

> Classical distributed-systems BFT встречается со stochastic LLMs. В 2025-2026 появились три research directions: **CP-WBFT** (arXiv:2511.10400) взвешивает каждый vote через confidence probe; **DecentLLMs** (arXiv:2507.14928) переходит к leaderless parallel worker proposals и geometric-median aggregation; **WBFT** (arXiv:2505.05103) сочетает weighted voting с Hierarchical Structure Clustering, чтобы разделять Core и Edge nodes. Честный empirical result из "Can AI Agents Agree?" (arXiv:2603.01213): даже scalar agreement сегодня хрупок — один deceptive agent может compromise Mixture-of-Agents. BFT necessary but not sufficient. Этот lesson строит minimal BFT protocol, injects три agent-specific attacks (byzantine lie, sycophantic conformity, correlated-error monoculture) и измеряет, как каждый consensus variant справляется.

**Тип:** Learn + Build
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Время:** ~75 minutes

## Цели обучения

- Объяснять, что дает классическая Byzantine fault tolerance, и три LLM-специфичных атаки.
- Реализовывать ядро протокола консенсуса в упрощенном виде.
- Читать ответы LLM-BFT 2025-2026 (CP-WBFT).

## Проблема

У вас есть N LLM agents, каждый производит answer. Они disagree. Majority vote выбирает неправильный вариант, потому что два agents correlated (same base model, same training data, same failure modes). Третий agent просто ошибается по-новому — и majority становится false majority.

Теперь добавьте deceptive agent: он намеренно lies. Или sycophantic agent: он agrees with whoever spoke last. В classical BFT предположение такое: Byzantine nodes составляют fraction `f < n/3` и behave arbitrarily. Реальность 2026: LLM nodes stochastic even when honest, correlated across models и influenced by each other's outputs. Нельзя treat them as independent Bernoulli voters.

Classical BFT (PBFT, 1999) не wrong — он incomplete. Он handles arbitrary bit-flipping. Он не handles "three honest agents share a hallucination because they share training data." Этот lesson строит от foundation PBFT и наслаивает три adaptations 2025-2026.

## Концепция

### What classical BFT gives you

Practical Byzantine Fault Tolerance (Castro & Liskov, OSDI 1999) tolerates `f < n/3` Byzantine nodes. Protocol имеет три phases (pre-prepare, prepare, commit) и два primitives (signed messages, quorum certificates). Agreement on a single value among `n >= 3f + 1` honest-or-malicious nodes.

Guarantees сильные, но предполагают:

1. **Independent faults.** Byzantines do not coordinate.
2. **Honest nodes are truly honest.** Correctness of honest outputs is a non-issue; protocol only aligns disagreement.
3. **The question has a ground-truth answer.** Consensus on a wrong fact is still consensus.

LLM agents нарушают все три. Два agents на одном base model share faults. "Honest" LLM все равно hallucinates. А в ambiguous questions "truth" — это то, что agents decide; external oracle нет.

### The three LLM-specific attacks

**Byzantine lie.** Один agent outputs deliberately wrong answer. Classical BFT handles this if `f < n/3`.

**Sycophantic conformity.** Один agent reads others' answers before voting и aligns with whoever spoke last. Не malicious, но correlates with the loudest voice. Classical BFT не prevents this, потому что agent passes every signature check.

**Correlated-error monoculture.** Три agents share a base model. Они hallucinate same wrong answer. Majority wrong. Classical BFT не помогает, потому что все трое "honestly" agree.

### The 2025-2026 responses

**CP-WBFT** (arXiv:2511.10400) — Confidence-Probed Weighted BFT. Каждый voter attaches confidence probe to its answer (self-reported probability или prediction отдельной calibration model). Vote weights scale with confidence. Reported +85.71% BFT improvement on complete graphs. Mitigation для: sycophantic conformity (conforming agents tend to have low confidence on their volunteered position).

**DecentLLMs** (arXiv:2507.14928) — Leaderless. Worker agents propose in parallel, evaluator agents score proposals, final answer is the geometric median of scored positions. Robust when `f < n/2`. Mitigation для: Byzantine lie и correlated errors (geometric median robust to outliers and pulls toward the dense cluster, not the model-biased average).

**WBFT** (arXiv:2505.05103) — Weighted BFT with Hierarchical Structure Clustering. Vote weights assigned by response quality plus trust score learned from history. Cluster agents into Core and Edge; Core agents must achieve consensus first, Edge agents follow. Mitigation для: scalability (Core consensus small and fast) и частично для monoculture (Core can be chosen for diversity).

### Empirical: "Can AI Agents Agree?" (arXiv:2603.01213)

Paper measures scalar agreement (LLM agents agreeing on a single numeric value) across multiple frontier models. Finding uncomfortable:

- Even with no adversaries, LLM agents disagree on scalar questions at rates above 30% on many benchmarks.
- A single agent that adopts a deceptive persona can pull the Mixture-of-Agents consensus 40+ percentage points off the honest baseline.
- Disagreement rates correlate with model diversity — heterogeneous ensembles disagree more than homogeneous ones (good: uncorrelated errors) but also drift more slowly (bad: longer time-to-agreement).

Takeaway: BFT дает machinery to align outputs, но не говорит, aligned output right ли. Combine with verification (Phase 16 · 08 role specialization), diversity (Phase 16 · 15 debate variants) и evaluator agents (Phase 16 · 24 benchmarks).

### The core protocol, stripped down

Minimal BFT round для LLM agents:

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

Semantic clustering step — LLM-specific twist. Два answers "the study reports 4.2%" и "4.2% improvement" — один и тот же cluster. Naive string-equality check пропустил бы это. В production используйте cheap embedding model или explicit canonicalization.

### Threshold tuning

Параметр `threshold` решает, когда accept, а когда retry. Too low: вы accept weak majorities. Too high: вы never accept anything. Empirical range: 0.5-0.67 для `n=5-7` agents, higher for smaller `n`. Below a threshold, escalate to a human or to a different agent ensemble.

### Where consensus does not help

- **Ambiguous questions.** Если question has no ground truth, consensus — opinion. Call it that.
- **Compound questions.** "Write code and explain it" — two answers. Vote on each independently.
- **Adversarial multi-round.** Если agents can observe prior rounds and mimic (Du 2023 debate), они start agreeing with each other regardless of truth. Bound the rounds (2-3 typically).

## Соберите

`code/main.py` реализует:

- `AgentVoter` — scripted policy with (answer, confidence).
- `MajorityVote` — classical plurality.
- `CPWBFT` — confidence-weighted voting with semantic clustering.
- `DecentLLMs` — geometric-median aggregation on scored proposals.
- `Scenario` — runs each aggregator under three attack patterns.

Attack patterns implemented:

1. `byzantine`: one agent lies with high confidence.
2. `sycophancy`: one agent copies the first answer it sees, with matching confidence.
3. `monoculture`: three agents share a wrong answer (correlated error) with moderate confidence.

Запуск:

```
python3 code/main.py
```

Ожидаемый вывод: table of (attack, aggregator) -> final answer, with correct answer highlighted. Plurality fails the monoculture case. CPWBFT confidence weighting mitigates sycophancy. DecentLLMs geometric-median pulls toward honest cluster when monoculture is less than half the population.

## Используйте

`outputs/skill-consensus-designer.md` проектирует consensus protocol для multi-agent ensemble: clustering method, weighting, threshold и escalation policy for sub-threshold rounds.

## Запустите в production

Перед выпуском любого consensus mechanism:

- **Attack-test with at least the three patterns** above. Protocol should fail predictably, not silently.
- **Log every minority cluster** with provenance. Minority clusters are your early-warning system for correlated errors.
- **Enforce bounded rounds.** No "keep debating until agreement" — that rewards sycophancy.
- **Separate agreement from correctness.** Consensus output goes to a verifier; verifier is independent of the ensemble.
- **Monitor the agreement rate.** Sharp rise means conformity bias; sharp fall means model drift.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что plurality fails monoculture attack, но CPWBFT partially mitigates it, когда monoculture confidence below 0.7.
2. Добавьте четвертый attack pattern: **silent abstention** — один agent отказывается отвечать ("I don't know"). Как каждый aggregator должен обрабатывать abstentions? Реализуйте ваш выбор.
3. Замените semantic clustering со string canonicalization на embedding-similarity (use any open-source embedding model). Что происходит с sycophancy attack?
4. Прочитайте CP-WBFT (arXiv:2511.10400). Реализуйте confidence-probe calibration step (separate calibration model checks each agent's self-reported confidence). Measure the accuracy gain on the monoculture scenario.
5. Прочитайте "Can AI Agents Agree?" (arXiv:2603.01213). Reproduce simplified scalar-agreement experiment: three agents, one scalar question, deceptive-persona prompt. Does CPWBFT or DecentLLMs catch it?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| BFT | "Byzantine fault tolerance" | Castro-Liskov 1999 protocol for consensus with `f < n/3` arbitrary faults. |
| Byzantine | "Any bad behavior" | Node, который может lie, drop messages, fail silently — anything but crash safely. |
| Confidence probe | "How sure are you?" | Self-reported or calibrator-predicted probability attached to a vote. |
| Semantic clustering | "Same answer, different words" | Grouping equivalent answers before counting votes. |
| Geometric median | "Robust center" | Point minimizing sum of distances to sample points. Robust to outliers, unlike mean. |
| Monoculture | "Same model, same failures" | Correlated errors when agents share training data or base model. |
| Sycophantic conformity | "Agreeing with the loud voice" | Vote agent biases toward whoever spoke first/loudest. |
| Core/Edge | "Hierarchical BFT" | WBFT split: small Core consensus first, Edge nodes follow. Bounds latency. |

## Дополнительное чтение

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — foundation
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — vote weighting by confidence
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — geometric-median aggregation
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — Core/Edge split for bounded latency
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — scalar-agreement fragility and deceptive-persona attack
