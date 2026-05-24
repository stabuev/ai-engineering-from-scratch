# SGLang и RadixAttention для prefix-heavy workloads

> SGLang рассматривает KV cache как first-class reusable resource, stored in a radix tree. Там, где vLLM schedules requests FCFS (first-come, first-served), cache-aware scheduler SGLang prioritizes requests with longer shared prefixes — фактически depth-first radix traversal, чтобы hot branches оставались resident in HBM. На Llama 3.1 8B с ShareGPT-like 1K prompts SGLang достигает ~16,200 tok/s против ~12,500 у vLLM, то есть edge ~29%. На prefix-heavy RAG workloads преимущество достигает 6.4x. На voice-cloning-shaped workloads cache hit rate превысил 86%. В 2026 году развернут на 400,000+ GPUs в xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS. Gotcha в том, что число 6.4x исчезает, когда prefix ordering inconsistent — ordering является рычагом инженера.

**Тип:** Изучение
**Языки:** Python (stdlib, учебный radix-tree cache + cache-aware scheduler)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**Время:** ~75 минут

## Цели обучения

- Нарисовать RadixAttention: как prefixes хранятся в radix tree и как KV blocks shared across sequences rooted at the same branch.
- Объяснить cache-aware scheduling и почему FCFS неправилен для prefix-heavy traffic.
- Посчитать expected speedup для workload по prefix-cache hit rate и prompt length distribution.
- Назвать prompt-ordering discipline, которая делает 6.4x реальным, а не lost upside.

## Проблема

Classic serving рассматривает prompt каждого request как opaque. Даже когда 5,000 RAG requests все начинаются с одного и того же 2,000-token system prompt плюс одинаковый retrieval preamble, vLLM prefills этот 2,000-token prefix 5,000 раз. GPU снова и снова делает одну и ту же работу.

Наблюдение: prompts в agentic и RAG workloads почти всегда разделяют длинные prefixes. System prompt, tool schemas, few-shot examples, retrieval headers, conversation history — все это повторяется между requests. Если хранить KV cache для такого prefix один раз и reuse, его не нужно prefill снова.

RadixAttention делает именно это. Tokens индексируются в radix tree; каждый node owns KV blocks for the token sequence on its path from root. Новый request проходит по дереву: любой node, token которого matches, re-uses KV blocks этого node. Prefill cost становится proportional to the "new" suffix, а не full prompt.

Сложность — scheduling. Если два requests делят 2,000-token prefix, а третий делит только 200 tokens того же prefix, нужно обслужить два long-shared requests вместе, чтобы long prefix оставался в HBM. FCFS делает обратное: обслуживает того, кто arrived first, потенциально evicting hot branch до того, как следующий long-prefix request попадет в cache.

## Концепция

### Radix tree как KV index

Radix tree (compact trie) хранит token sequences. Каждый node owns token range and the KV blocks computed for that range. Children extend the sequence one or more tokens.

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

Новый request приходит с system prompt + "Context: <doc A>" + "Question: Carol". Scheduler walks: system prefix matches (124 blocks reused), doc-A branch matches (31 blocks reused), затем allocates fresh blocks только для "Question: Carol" (4 blocks). Prefill cost: 4 blocks of new tokens. Без дерева: 160 blocks. ~40x savings on prefill.

### Cache-aware scheduling

Radix-tree-backed reuse бесполезен, если cache churns. Две ключевые policies:

1. **Depth-first dispatch**. Выбирая следующий request из queue, предпочитайте requests, rooted at the same branch as the current running set. Это keeps the hot branch pinned.
2. **LRU at branch level, not block level**. Evict whole branches (starting from shortest-used leaves), а не individual blocks, чтобы cache shape соответствовал radix shape.

FCFS нарушает оба правила. Request, sharing 2,000 tokens, сидит за request, sharing 50, затем 2,000-token branch evicted, чтобы принять 50-token one.

### Benchmark numbers, которые нужно запомнить

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16,200 tok/s vs vLLM ~12,500 (~29% edge).
- Prefix-heavy RAG (same system + same doc, varying question): up to 6.4x on SGLang.
- Voice cloning workloads: 86.4% prefix-cache hit rate.
- Production hit rates across SGLang customers: 50-99% depending on prompt discipline.
- Deployed on 400,000+ GPUs in 2026.

### Gotcha ordering

Число 6.4x зависит от consistent prompt-template ordering. Если client строит prompts как `[system, tools, context, history, question]` в одних requests и `[system, context, tools, history, question]` в других, tree не может найти shared prefix. То, что человеку кажется shared prefix, для radix tree — две разные sequences.

Рычаг инженера: prompt template — это cache key. Зафиксируйте порядок. Все immutable (system, tools, schemas) ставьте первым. Retrieval context ставьте следующим. User question — последним. Не interleave dynamic content into the prefix.

Реальный случай из research: перенос dynamic content из cacheable prefix поднял один deployment с 7% до 74% cache hit rate одним изменением.

### Где RadixAttention выигрывает и проигрывает

Выигрывает:
- RAG (same retrieval preamble, varying question).
- Agents (same tool schemas, varying query).
- Chat with long system prompt.
- Voice / vision workloads with repeated preambles.

Проигрывает (возвращается к vLLM-level throughput):
- Single-shot generation with unique prompts (code completion, open-ended chat without system prompt).
- Dynamic prompts, где every request interleaves unique content into the prefix.

### Почему это scheduler problem, а не только kernel problem

Можно реализовать KV reuse как kernel trick. Insight SGLang в том, что reuse окупается только если scheduler keeps the hot branch resident. Наивная policy "reuse if available" будет churn cache under mixed load. Radix-tree-indexed scheduler превращает kernel trick в 29% production edge.

### Interplay with vLLM

Две системы не являются строгими competitors. В 2026 году vLLM добавил prefix caching (`--enable-prefix-caching`) и cache-aware router (vLLM Router in Rust). Gap сократился, но не исчез полностью: SGLang whole stack is radix-first; vLLM grafted it on. Для workloads, dominated by prefix reuse, SGLang остается default. Для general-purpose serving без strong prefix patterns vLLM остается равным или лучше.

## Используйте это

`code/main.py` реализует учебный radix-tree KV cache плюс scheduler с двумя policies: FCFS и cache-aware. Прогоняет один workload через оба, выводит prefix-cache hit rate и throughput delta. Затем запускает "scrambled ordering" workload, чтобы показать collapse 6.4x.

## Доведите до результата

Этот урок создает `outputs/skill-radix-scheduler-advisor.md`. По workload description (prompt-template shape, retrieval pattern, number of concurrent tenants) он выдает prompt-ordering prescription и go/no-go for SGLang adoption.

## Упражнения

1. Запустите `code/main.py`. Сравните FCFS и cache-aware на одном workload. Откуда берется delta: prefill savings, decode savings или queue delay?
2. Измените workload так, чтобы prompts randomly permute `[system, tools, context]`. Запустите снова. Что происходит с hit rate? Почему?
3. Посчитайте HBM cost хранения 2,000-token system prompt resident как one radix branch на Llama 3.1 8B. Сравните с cost of a 16-sequence batch without prefix reuse.
4. Прочитайте SGLang RadixAttention paper. Объясните в трех sentences, почему tree-shaped LRU eviction лучше block-shaped LRU under prefix-heavy load.
5. Customer reports only 8% cache hit rate. Назовите три вероятные причины и diagnostic, который вы запустите для каждой.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| RadixAttention | "the SGLang thing" | KV cache indexed as a radix tree so shared prefixes reuse blocks |
| Radix tree | "compact trie" | Tree, где каждый node owns a token range and its KV blocks |
| Cache-aware scheduler | "hot-branch-first" | Scheduler, который prefers requests sharing the resident branch |
| Prefix-cache hit rate | "how much of your prompt was free" | Доля prompt tokens served from reused KV blocks |
| FCFS | "first-come first-served" | Default scheduling, которое breaks prefix locality |
| Branch-level LRU | "evict the leaf" | Eviction policy, matched to radix shape |
| Prompt template ordering | "the cache key" | Порядок компонентов prompt определяет, что tree can share |
| System prompt pinning | "resident prefix" | Keep the immutable system portion pinned to avoid eviction thrash |

## Дополнительное чтение

- [SGLang GitHub](https://github.com/sgl-project/sglang) — source and docs.
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention and scheduling details.
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — design reference.
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — benchmark numbers and scheduler rationale.
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM's own radix-like implementation, for comparison.
