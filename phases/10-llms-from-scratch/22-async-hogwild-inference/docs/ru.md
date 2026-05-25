# Async and Hogwild! Inference

> Speculative decoding (Phase 10 · 15) parallelizes tokens within one sequence. Multi-agent frameworks parallelize across whole sequences, но требуют explicit coordination (voting, sub-task splitting). Hogwild! Inference (Rodionov et al., arXiv:2504.06261) делает другое: запускает N instances одной LLM параллельно против SHARED key-value cache. Каждый worker мгновенно видит tokens, сгенерированные другими workers. Современные reasoning models — QwQ, DeepSeek-R1 — могут self-coordinate через shared cache без fine-tuning. Подход экспериментальный, но он открывает совершенно новую ось inference parallelism, ортогональную spec decode. Этот урок реализует two-worker Hogwild! simulator на stdlib Python и объясняет, почему shared-cache collaboration возникает из уже существующих reasoning abilities модели.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Цели обучения

- Описать три common parallel-LLM topologies (voting, sub-task, Hogwild!) и назвать, какие problems решает каждая.
- Сформулировать core Hogwild! setup: multiple workers, one shared KV cache, emergent coordination via self-prompting.
- Посчитать wall-time speedup Hogwild! как функцию worker count `N`, task-level parallelism `p` и coordination overhead `c`.
- Реализовать two-worker Hogwild! simulator на toy problem и наблюдать emergent task division.

## Проблема

Современные LLMs решают сложные problems, производя длинные chains of reasoning — 5000 tokens пошаговой логики обычны, десятки тысяч tokens встречаются на глубоких math problems. При 35 tokens/sec decode на 70B model 50k tokens — это 24 минуты. Interactive model такой не является.

Speculative decoding (Phase 10 · 15) дает 3-5x speedup за счет parallelizing within one sequence. Дальше жесткий потолок — sequential dependency autoregressive decoding. Каждый новый token зависит от всех предыдущих.

Очевидный вопрос: можно ли parallelize across sequences? Запустить несколько copies одной model на одной problem, позволить им cooperate, дать им divide the work?

Previous work: voting ensembles (запустить N models, выбрать majority answer), tree-of-thought (branch reasoning paths and recombine), multi-agent frameworks (назначить каждому agent sub-task, использовать coordinator). Все это помогает в specific task domains. И все это вводит explicit coordination machinery — voting rules, branch-and-prune logic, agent-to-agent messaging protocols.

Hogwild! Inference выбирает другой подход. N workers разделяют один KV cache. Каждый worker мгновенно видит tokens всех остальных workers, как будто это его собственный context. Workers — без training или fine-tuning — выясняют, как divide the work. Современные reasoning models (QwQ, DeepSeek-R1, Claude-family reasoning mode) могут читать shared cache и писать вроде: "I see worker 2 already handled the base case, so I'll work on the inductive step."

Speedup workload-dependent и экспериментальный по состоянию на апрель 2026. Но идею нужно знать, потому что она открывает новую ось inference parallelism.

## Концепция

### The setup

Инициализируйте N worker processes, все запускают одну и ту же LLM. Вместо per-worker KV caches поддерживайте ONE shared cache. Когда worker `i` генерирует token `t_j`, token записывается в shared cache на следующую position. Когда worker `k` делает следующий step, он читает current state cache (включая все, что сгенерировали все N workers).

В step time workers соревнуются за запись tokens. Per-worker position index нет — cache является одной растущей sequence. Order определяется write arrival time.

### Почему возникает координация

Workers используют общий prompt. Обычно это что-то вроде "You are one of N instances working together on this problem. Each instance reads the shared memory and can see what other instances have written. Avoid redundant work." Prompt плюс shared cache обычно достаточно. Reasoning models читают cache, замечают, какие части задачи уже были опробованы, и часто переключаются на еще не исследованные направления.

Hogwild! paper (Rodionov et al., 2025) сообщает observations:

- Workers формулируют plans и передают их другим workers через cache.
- Workers замечают errors в reasoning других workers и явно указывают на них.
- Workers adapt, когда plan fails, и предлагают alternatives.
- Когда prompted to check for redundancy, workers detect it and pivot.

Ничего из этого не требует fine-tuning. Emergent behavior приходит из reasoning capabilities, которые у model уже есть.

### The naming

Название paper отсылает к Hogwild! SGD (Recht et al., 2011), asynchronous-update optimizer. Аналогия: asynchronous workers SGD все пишут в shared parameter vector; workers Hogwild! Inference все пишут в shared KV cache. Оба полагаются на empirical convergence, а не synchronization guarantees.

### RoPE makes this tractable

Rotary Position Embeddings (RoPE, Su et al. 2021) кодируют position information через rotation в Q и K vectors. Поскольку positions — rotations, а не baked-in offsets, position token может сдвигаться без recomputing KV cache entry. Когда worker `i` пишет в shared cache на position `p`, другие workers, читающие эту position, могут использовать cached entry напрямую — re-rotation не нужна.

В learned-position или absolute-position model Hogwild! требовал бы cache invalidation при каждой concurrent write. RoPE позволяет cache оставаться stable.

### Wall-time math

Пусть `T_serial` — время, за которое один worker решает problem alone. Пусть `p` — task-level parallelizable fraction. Пусть `c` — per-step coordination overhead (reading extended cache, deciding what to write).

Single-worker time: `T_serial`.
N-worker Hogwild! time, если coordination free: `T_serial * ((1 - p) + p / N)`. Классический Amdahl.
С coordination overhead: `T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

Чтобы worker был productive, `c` должен быть мал относительно per-step decode time. На reasoning models, производящих 5k+ tokens, workers могут позволить себе сотни tokens coordination overhead и все равно выиграть. На short chat tasks coordination dominates, и Hogwild! хуже serial.

### Concrete example

Reasoning problem: 10k tokens chain-of-thought. Пусть problem имеет `p = 0.7` parallelizable content (different proof strategies, different case analyses) и `c = 200` tokens coordination overhead на worker. С `N = 4` workers:

- Serial time: 10000 decode steps.
- Hogwild! time: 10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 decode steps.
- Speedup: 10000 / 5550 = 1.8x.

Это умеренно. Но на более длинных reasoning problems (50k tokens) coordination overhead амортизируется, и speedup выходит к 2.5-3x. Hogwild! — inference equivalent thread-level parallelism в языке, где natural multi-threaded code пишется естественно.

### When to reach for Hogwild!

- Long reasoning problems (тысячи tokens), где task можно parallelize across independent sub-goals.
- Reasoning models, обученные think step by step. Non-reasoning models плохо self-coordinate.
- Single-node deployments с достаточной VRAM для shared cache плюс N worker processes. Cache shared, но у каждого worker своя activation memory.

### When not to

- Short interactive chat. Coordination overhead dominates.
- Tasks that don't parallelize (single linear proof, single compilation). N=1 — максимум.
- Non-reasoning models. Coordination не возникает.
- Multi-node deployments. Shared cache требует очень fast cross-worker synchronization. Intra-node нормально; cross-node — latency disaster.

### The experimental status

По состоянию на апрель 2026 Hogwild! — research method с open-source PyTorch implementation. Production adoption еще не случился. Три blockers:

1. Shared KV cache management across concurrent processes — нетривиальная engineering задача.
2. Emergent coordination task-dependent; benchmarks still being built.
3. Speedups modest compared to what speculative decoding already delivers, and the two can be combined but the combined engineering is another layer.

Стоит знать. Стоит экспериментировать. Пока не стоит ставить на него product.

## Практика

`code/main.py` реализует toy Hogwild! simulator:

- Two worker processes, каждый deterministic "LLM", производящий одну из нескольких token categories (work-token, observe-token, coordinate-token) с известными probabilities.
- Shared cache (просто list of tokens), который оба workers read and write.
- Простая coordination logic: когда worker видит, что другой уже произвел достаточно work tokens в category, он выбирает другую category.

Simulator runs for fixed step budget и reports:

- Total work-tokens produced.
- Total wall time (number of worker steps).
- Effective speedup over a single worker.
- Trace of which worker wrote which token.

### Step 1: the shared cache

List, в который оба workers append. Simple locking (Python `threading.Lock`) в real implementation; здесь мы simulate with a counter.

### Step 2: the worker loop

Каждый worker на каждом step:

- Reads current shared cache.
- Decides what category of token to write based on what is already there.
- Writes one token.

### Step 3: the coordination heuristic

Если category X уже имеет K tokens в cache и intended category worker равна X, worker switches to category Y. Это toy stand-in для behavior reasoning-model: "notice this is already covered, do something else instead."

### Step 4: measured speedup

Запустите simulator с N=1 worker и N=2 workers при том же total step budget. Посчитайте work-tokens produced. N=2 должен производить примерно в 1.5-1.8x больше work-tokens благодаря coordination-driven task division.

### Step 5: stress the coordination

Уменьшите sensitivity coordination heuristic. Запустите снова. Наблюдайте, что без хорошей coordination N=2 redundantly produces same tokens и speedup падает ниже 1. Это совпадает с observation paper: trick работает только если workers имеют reasoning capacity для self-coordinate.

## Использование

Hogwild! integration in production по состоянию на апрель 2026 — research-grade. Reference implementation от Yandex/HSE/IST основана на PyTorch и нацелена на single-node multi-process setups на DeepSeek-R1 и QwQ models.

Практичный путь внедрения:

1. Профилируйте workload для reasoning-задач. Измерьте долю tokens, которые являются exploratory (multiple strategies, case analyses, search), по сравнению с linear.
2. Если exploration dominates, запустите two-worker Hogwild! experiment. Измерьте wall-time improvement.
3. Если improvement меньше 1.3x, вы в coordination-dominated regime. Revert to single-worker.
4. Если improvement выше 1.5x, попробуйте N=4 и измерьте снова. Diminishing returns обычно возникают около N=4-8.

Combine with speculative decoding: каждый Hogwild! worker может independently use spec decode. Два speedups умножаются (roughly), превращая 3x spec decode и 1.8x Hogwild! в effective 5.4x относительно naive single-worker decoding.

## Результат

Этот урок создает `outputs/skill-parallel-inference-router.md`. По reasoning workload profile (token budget, task parallelism profile, model family, deployment target) он routes between voting, tree-of-thought, multi-agent, Hogwild!, and speculative decoding strategies.

## Упражнения

1. Запустите `code/main.py` с default settings. Подтвердите, что N=2 Hogwild! configuration производит больше work-tokens, чем N=1 baseline за то же wall time.

2. Уменьшите strength coordination heuristic (set `coordination_weight=0.1`). Запустите заново. Покажите, что speedup collapses. Объясните почему: workers duplicate effort, когда не могут coordinate.

3. Посчитайте expected Hogwild! speedup для 50k-token reasoning task с `p=0.8, c=500` и N=4 workers. Сделайте то же для 1k-token chat task с `p=0.3, c=200` и N=4. Почему в одном случае win, а в другом loss?

4. Прочитайте Section 4 paper Hogwild! (preliminary evaluation). Назовите два failure modes, которые report authors. Опишите, как better coordination prompt мог бы смягчить каждый.

5. Combine Hogwild! with speculative decoding в toy: каждый worker использует 2-token spec-decode internally. Report multiplicative speedup. Какая bookkeeping problem возникает, когда два workers оба хотят extend same shared-cache prefix?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Hogwild! | "Parallel workers, shared cache" | N instances одной LLM running concurrently с одним shared KV cache; emergent coordination via self-prompting |
| Shared KV cache | "The coordination medium" | Один растущий KV buffer, который все workers read and write; дает instant token visibility across workers |
| Emergent coordination | "No training needed" | Reasoning-capable LLMs могут читать shared cache и divide work без fine-tuning или explicit protocol |
| Coordination overhead (c) | "Tokens spent orienting" | Per-worker cost чтения extended cache и решения, что делать; должен оставаться малым vs total decode time |
| Parallelizable fraction (p) | "What can run in parallel" | Task-level parallelism: fraction total work, которая intrinsically sequential |
| RoPE enables Hogwild! | "Rotary positions are shift-invariant" | Поскольку positions — rotations, запись в shared cache не требует recomputing prior tokens |
| Voting ensemble | "Run N, pick the majority" | Простейшая parallel inference topology; полезна для classification, меньше для long-form reasoning |
| Tree of thought | "Branch and prune" | Reasoning strategy, explores multiple branches and prunes; explicit coordination logic |
| Multi-agent framework | "Assign sub-tasks" | Каждый agent получает role; coordinator orchestrates; heavy protocol overhead |

## Дополнительное чтение

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — paper Hogwild!, preliminary evaluation на QwQ и DeepSeek-R1
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — original Hogwild!, origin naming
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, property that makes shared-cache inference tractable
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — tree-of-thought reasoning strategy, orthogonal to Hogwild!
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative decoding, within-sequence parallelism, with which Hogwild! composes
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — single source of truth for paper experiments
