# Voting, Self-Consistency, and Debate Topology

> Самая дешевая aggregation: sample N independent agents, majority-vote. Wang et al. 2022 self-consistency делал это с одной model, sampled N times. Multi-agent расширяет это **heterogeneous** agents, чтобы уйти от monoculture — разные models, разные prompts, разные temperatures, разные contexts. За пределами majority vote важна debate topology: MultiAgentBench (arXiv:2503.01935, ACL 2025) оценил star / chain / tree / graph coordination и found **graph best for research**, с "coordination tax" после ~4 agents. AgentVerse (ICLR 2024) документирует два emergent patterns — volunteer behaviors и conformity behaviors — и conformity одновременно feature (finding consensus) и risk (groupthink, Lesson 24). Этот lesson maps topology space, builds each variant и measures coordination tax.

**Тип:** Learn + Build
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Время:** ~75 minutes

## Проблема

Debate может improve accuracy (Du et al., arXiv:2305.14325). Он также может degrade it. Поможет ли debate, зависит от четырех structural choices:

1. Who talks to whom (topology).
2. Сколько rounds (Du 2023: both rounds and agents matter independently).
3. Whether agents are heterogeneous (different base models break monoculture).
4. Whether an adversarial voice is present (steel-manning vs. straw-manning).

Teams, которые просто прикручивают "run 5 agents and vote" к task, часто regress vs. single agent. Failures не random. Они track topology and heterogeneity. Этот lesson — topology map.

## Концепция

### Self-consistency, the single-model baseline

Wang et al. 2022 ("Self-Consistency Improves Chain of Thought Reasoning") sampled the same model N times at temperature > 0 and majority-voted on reasoning-path answers. Result on GSM8K: substantial gains with N=40 samples over a single greedy decode. Self-consistency — single-agent precursor to multi-agent voting.

Limit: self-consistency uses one base model. Errors are correlated by construction. Если model имеет systematic bias, все N samples share it.

### Multi-agent vote, the heterogeneous extension

Replace N samples with N *different* agents. Different base models (Claude, GPT, Llama), different prompts, different tool access. Benefit: uncorrelated errors. Cost: different agents cost different amounts; coordinating them adds overhead.

Canonical 2026 name для heterogeneous debate — **A-HMAD**: Adversarial Heterogeneous Multi-Agent Debate. Термин принят не везде, но papers используют его для "different models debate, which reduces correlated errors from monoculture collapse."

### The four topologies

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star: one hub, all others talk only to hub. Equivalent to supervisor-worker without back-channel.
Chain: linear, each agent sees the prior one's output. Pipeline-like.
Tree: hierarchical, used by hierarchical agent systems (Lesson 06).
Graph: any-to-any. Includes fully-connected clique and arbitrary DAGs.

### The coordination tax (MultiAgentBench)

MultiAgentBench (MARBLE, ACL 2025, arXiv:2503.01935) benchmarked star, chain, tree, graph on a task suite including research, coding, and planning. Key measured results:

- **Graph** topology wins on research tasks. Information flows any-to-any; agents can critique each other.
- **Star** wins on fast-answer factual tasks. Hub filters and consolidates.
- **Chain** wins on stepwise pipelines (staged refinement).
- **Coordination tax** appears past ~4 agents in graph topology. Wall-clock and token cost grow faster than quality.

4-agent ceiling empirical, not fundamental. Он отражает context capacity LLM 2026: context каждого agent заполняется outputs peers, а marginal value добавления agent N+1 падает, когда everyone can see everyone.

### Multi-Agent Debate Strategies ("Should we be going MAD?")

arXiv:2311.17371 — survey 2023 по MAD strategies. Key finding, replicated by others: MAD variants, которые *structurally similar* to self-consistency (independent sampling + aggregation), часто underperform self-consistency при same budget. MAD helps most, когда agents genuinely heterogeneous и debate имеет adversarial structure (one agent argues against).

### AgentVerse emergent patterns

AgentVerse (ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) документирует два behaviors, которые emerge from multi-agent debate even without explicit design:

- **Volunteer.** Agent предлагает help ("I can take the next step") unprompted. Useful: allocates work to the most-capable agent for a subtask.
- **Conformity.** Agent adjusts its stance to match a critic, even when critic is wrong. Это debate-equivalent of sycophancy (Lesson 14).

Conformity — причина, почему debate-until-agreement rewards bullies. Bounded rounds with a separate judge mitigate.

### Heterogeneity: the actual knob that moves accuracy

Pattern 2024-2026 в practical literature: замена одного из N agents на different base model дает bigger accuracy bump, чем increasing N by 1. Intuition is monoculture — each new independent-error source worth more than additional correlated sample.

In the limit, heterogeneity beats numerosity. Three different models beat five copies of one model on most tasks that have clean ground truth.

### Jury methods

Sibyl framework (cited in Minsky-LLM literature) formalizes a "jury" — small set of specialized agents that refine answers by voting at each stage. Unlike plain majority vote, jury has roles: one agent cross-examines, one supplies context, one scores plausibility. Jury methods are midpoint between plain vote (cheap, monoculture-prone) and full MAD (expensive, conformity-prone).

### When vote-with-debate dominates

- Question has ground truth (fact, math, code behavior). Vote convergence is meaningful.
- Agents can access different sources or tools (heterogeneity is available).
- Rounds are bounded (2-3 typical) and there is a separate judge or verifier.
- Budget allows 3-5 agents. Beyond 5-7 on graph topology, coordination tax dominates.

### When vote-with-debate hurts

- Вопрос имеет форму мнения. Агенты сходятся к ответу, который выглядит самым уверенным, а не самым правильным.
- All agents share a base model. Monoculture makes consensus meaningless.
- Rounds are unbounded. Conformity wins every time.
- Task is simple. A single agent with self-consistency at N=5 is cheaper and as accurate.

## Соберите

`code/main.py` реализует:

- `run_star(agents, hub, question)` — hub polls each worker, aggregates.
- `run_chain(agents, question)` — sequential refinement.
- `run_tree(root, children, question)` — hierarchical with depth-2 aggregation.
- `run_graph(agents, question, rounds)` — all-to-all debate, bounded rounds.
- Scripted heterogeneity dial: у каждого agent есть `error_bias`, indicating its systematic wrongness.
- Measurement harness, который runs each topology at N=3, 5, 7 и reports (accuracy, total_tokens, wallclock_simulated).

Запуск:

```
python3 code/main.py
```

Ожидаемый вывод: table of topology × N → (accuracy, tokens, latency). Graph wins at N=3-5 on research-style tasks; star wins on fast-factual tasks; graph at N=7 shows coordination tax (latency inflates faster than accuracy).

## Используйте

`outputs/skill-topology-picker.md` — skill, который reads task description and recommends topology (star / chain / tree / graph), N (number of agents), heterogeneity profile (base models to use) и round bound.

## Запустите в production

For any ensemble:

- Start with **self-consistency at N=5** using one strong base model. It is the cheap baseline.
- Upgrade to **heterogeneous voting at N=3** if accuracy matters. Measure the delta.
- Only upgrade to **debate topology** if the task has structure (research, multi-step) and bounded rounds are feasible.
- Всегда логируйте minority cluster. Когда minority стабильно права, у вас есть diversity signal.
- Benchmark wall-clock and tokens alongside accuracy. "Better accuracy at 10x cost" is a business decision.

## Упражнения

1. Запустите `code/main.py`. Постройте coordination-tax curve для graph topology: accuracy vs N, tokens vs N. At what N does the curve inflect?
2. Реализуйте A-HMAD: three agents с намеренно разными biases. Как all-same-bias baseline сравнивается с A-HMAD на monoculture attack из Lesson 14?
3. Добавьте роль "judge" в graph topology, которая does not vote, only scores the final consensus. Does this change the emergent conformity behavior?
4. Прочитайте AgentVerse paper (ICLR 2024). Identify which emergent behavior your implementation exhibits most strongly. Can you elicit the opposite behavior by a prompt change?
5. Прочитайте MultiAgentBench (arXiv:2503.01935) Section 4 (topology experiments). Reproduce the "graph-wins-research" result on one task from the paper using your harness.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Self-consistency | "Sample N times, vote" | Wang 2022. Single model, N temperature>0 samples, majority vote on reasoning paths. |
| Heterogeneity | "Different models" | Ensemble of different base models or prompt families. Breaks monoculture. |
| MAD | "Multi-agent debate" | Generic term for agents exchanging critiques over rounds. See Du 2023. |
| A-HMAD | "Adversarial Heterogeneous MAD" | MAD variant emphasizing different models + adversarial structure. |
| Topology | "Who talks to whom" | Star, chain, tree, graph. Determines information flow. |
| Coordination tax | "Diminishing returns" | Above ~4 agents on graph, cost grows faster than quality. |
| Volunteer behavior | "Unprompted help" | AgentVerse emergent pattern: agent offers to take a step. |
| Conformity behavior | "Agreement under pressure" | AgentVerse emergent pattern: agent aligns with a critic. |
| Jury | "Small specialized panel" | Sibyl-style ensemble with roles (examiner, context, scorer). |

## Дополнительное чтение

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — single-model baseline
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — both agents AND rounds matter independently
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark showing graph best for research, chain for pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD-strategy survey; finds MAD often loses to self-consistency at equal budget
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer and conformity emergent patterns
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — reference benchmark implementation
