# Disaggregated Prefill/Decode — NVIDIA Dynamo and llm-d

> Prefill является compute-bound; decode является memory-bound. Запуск обоих на одной GPU растрачивает один из ресурсов. Disaggregation разделяет их на разные pools и передает KV cache между ними через NIXL (RDMA/InfiniBand или TCP fallback). NVIDIA Dynamo (анонс на GTC 2025, 1.0 GA) находится над vLLM/SGLang/TRT-LLM — ее Planner Profiler + SLA Planner автоматически подбирают prefill:decode ratios для выполнения SLOs. NVIDIA публикует прирост throughput примерно такого масштаба — developer.nvidia.com (2025-06) показывает ~6x improvement для DeepSeek-R1 MoE на GB200 NVL72 + Dynamo в medium-latency regime, а product page Dynamo (developer.nvidia.com, без даты) заявляет до 50x MoE throughput на GB300 NVL72 + Dynamo vs Hopper. Цифра "30x" — community aggregate по full-stack Blackwell + Dynamo + DeepSeek-R1 reports; мы не нашли единого primary source, который утверждает ровно 30x, поэтому относитесь к ней как к directional claim. llm-d (Red Hat + AWS) Kubernetes-native: prefill / decode / router как независимые Services с per-role HPA. llm-d 0.5 добавляет hierarchical KV offloading, cache-aware LoRA routing, UCCL networking, scale-to-zero. Economics: внутренний rollup нескольких customer disclosures предполагает 30–40% savings на inference spend класса $2M (то есть $600-800K/year) при переходе с colocated serving на disaggregated with Dynamo при constant SLA; конкретная цифра $2M→$600-800K — internal composite, не один опубликованный case study — используйте ее как order-of-magnitude anchor, а не reference citation. Короткие prompts (<512 tokens, short output) не оправдывают transfer cost.

**Тип:** Learn
**Языки:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**Время:** ~75 minutes

## Цели обучения

- Объяснить, почему prefill и decode имеют разные оптимальные GPU allocations, и количественно оценить waste при colocation.
- Нарисовать disaggregated architecture: prefill pool, decode pool, KV transfer via NIXL, router.
- Назвать условие, когда disaggregation НЕ окупается (short prompts, short outputs).
- Отличить NVIDIA Dynamo (stack-above) от llm-d (Kubernetes-native) и сопоставить каждую с operational context.

## Проблема

Вы запускаете Llama 3.3 70B на 8 H100. При mixed workload (long prompts + short outputs) GPU простаивают во время decode, потому что большая часть compute была потрачена на prefill. При другой workload (short prompts + long outputs) происходит обратное. Colocated prefill + decode означает, что вы over-provision оба ресурса.

Влияние на бюджет: 20-40% GPU time тратится на неправильный ресурс. Вы покупаете H100 compute, чтобы выполнять memory-bound decode, или покупаете H100 HBM bandwidth, чтобы выполнять compute-bound prefill. И то и другое — дорогой waste.

Disaggregation разделяет prefill и decode по отдельным pools, размер которых соответствует bottleneck каждого этапа. KV cache передается из prefill pool в decode pool через high-bandwidth interconnect.

## Концепция

### Why the bottlenecks differ

**Prefill** — прогон transformer по всему input prompt за один forward. Доминируют matrix multiplications; compute-bound. H100 FP8 дает ~2000 TFLOPS полезного throughput. Batch efficiency хорошая — один forward обрабатывает много tokens.

**Decode** — генерирует по одному token за раз, читая все weights на каждой итерации. Memory-bandwidth-bound. HBM3 дает ~3 TB/s. Batch efficiency хорошая только при высокой concurrency — чтение weights амортизируется по batch.

При colocation вы покупаете GPUs, оптимизированные для обоих режимов. H100 хороша в обоих, но стоит одинаково в любом случае. В масштабе нужен prefill pool на H100 / compute-heavy; decode pool на H200 / memory-heavy, или с aggressive quantization.

### The architecture

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL — inter-node transport NVIDIA. Использует RDMA/InfiniBand, когда доступно, иначе TCP fallback. Transfer latency реальна — обычно 20-80 ms для KV cache prompt на 4K tokens на 70B FP8. Поэтому short prompts не оправдывают disaggregation: transfer tax превышает savings.

### Dynamo vs llm-d

**NVIDIA Dynamo** (анонс на GTC 2025, 1.0 GA):
- Находится над vLLM, SGLang, TRT-LLM как orchestrator.
- Planner Profiler измеряет workload, SLA Planner автоматически конфигурирует prefill:decode ratios.
- Rust core, Python extensibility.
- Throughput gains: NVIDIA сообщает 6x для DeepSeek-R1 MoE на GB200 NVL72 + Dynamo в medium-latency regime (developer.nvidia.com, 2025-06); community reports "up to 30x" на full Blackwell + Dynamo + DeepSeek-R1 stacks не имеют единого primary source и должны восприниматься как directional.
- GB300 NVL72 + Dynamo: до 50x MoE throughput vs Hopper по product page Dynamo (developer.nvidia.com, без даты).

**llm-d** (Red Hat + AWS, Kubernetes-native):
- Prefill / decode / router как независимые Kubernetes Services.
- Per-role HPA с signals queue depth (prefill) / KV utilization (decode).
- `topologyConstraint packDomain: rack` упаковывает prefill+decode cliques в один rack для high-bandwidth KV transfer.
- llm-d 0.5 (2026): hierarchical KV offloading, cache-aware LoRA routing, UCCL networking, scale-to-zero.

Используйте Dynamo, если нужен managed stack-above orchestrator. Используйте llm-d, если нужны Kubernetes-native primitives и вы привязаны к CNCF ecosystem.

### Economics

Internal composite (не единый опубликованный case study — order-of-magnitude anchor):

- $2M/year inference spend на colocated serving.
- Переход на disaggregated with Dynamo.
- Тот же request volume, тот же P99 latency SLA.
- Reported savings: $600K–$800K/year (30–40% reduction).
- Без нового hardware.

Мы синтезируем эту цифру из нескольких customer disclosures, а не из одного citable case study; ближайший опубликованный datapoint — Baseten's 2x faster TTFT / 61% higher throughput with Dynamo KV routing (baseten.co, 2025-10), и VAST + CoreWeave projection of 60–130% more tokens/$ at 40–60% KV hit rate (vastdata.com, 2025-12). Savings возникают из right-sizing каждого pool; prefill-heavy workloads (RAG with 8K+ prefixes) выигрывают больше, чем balanced.

### When NOT to disaggregate

- Prompts < 512 tokens и outputs < 200 tokens: transfer tax dominates gain.
- Small cluster (< 4 GPUs): недостаточно pool diversity.
- Команда не может эксплуатировать two GPU pools with per-role scaling: Dynamo помогает, но не делает это trivial.
- Нет RDMA fabric: TCP transfer tax тяжелее.

### The router integrates with Phase 17 · 11

Disaggregated routers являются KV-cache-aware (Phase 17 · 11). Request попадает на decode pool, где хранится его prefix; если match нет, он идет prefill → decode. Hit rate и disaggregation усиливают друг друга — cache-aware router определяет, нужен ли новый prefill вообще.

### MoE on Blackwell is where the real numbers are

GB300 NVL72 + Dynamo показывает 50x MoE throughput относительно Hopper baselines. MoE expert routing compute-heavy на prefill, но memory-heavy на decode (expert caches), поэтому disaggregation дает двойной выигрыш. Serving frontier models в 2026 году MoE-dominant (DeepSeek-V3, будущие варианты GPT-5).

### Numbers you should remember

Benchmark numbers меняются — NVIDIA и inference stack публикуют обновленные результаты каждый квартал. Перепроверяйте перед цитированием.

- DeepSeek-R1 on GB200 NVL72 + Dynamo: ~6x throughput vs baseline in the medium-latency regime (developer.nvidia.com, 2025-06); community "up to 30x" claims on full Blackwell + Dynamo stacks are directional aggregates without a single primary source.
- GB300 NVL72 + Dynamo: up to 50x MoE throughput vs Hopper (developer.nvidia.com, undated).
- Savings anchor (internal composite, not a single case study): $600-800K/year off a $2M annual spend at constant SLA.
- Disaggregation threshold: prompts >512 tokens + outputs >200 tokens.
- KV transfer via NIXL: 20-80 ms for 4K-prompt KV on 70B FP8.

## Используйте это

`code/main.py` симулирует colocated vs disaggregated serving. Показывает throughput, cost per request и prompt-length crossover.

## Доведите до результата

Этот урок создает `outputs/skill-disaggregation-decider.md`. По workload и cluster он решает, стоит ли делать disaggregate.

## Упражнения

1. Запустите `code/main.py`. При какой prompt length disaggregation выигрывает у colocation?
2. Спроектируйте prefill pool и decode pool для RAG service с P99 prefix length 8K, output 300.
3. Dynamo vs llm-d: выберите один для pure-Kubernetes shop без preference к Python runtime.
4. Посчитайте KV transfer cost: 4K prefill on 70B FP8 = ~500 MB KV. At RDMA 100 GB/s, transfer = 5 ms. At TCP 10 GB/s = 50 ms. Что важно для вашего SLA?
5. MoE expert routing меняет KV access patterns. Как disaggregation ведет себя с MoE, который activates different experts per token?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Disaggregated serving | "split prefill/decode" | отдельные GPU pools для каждой фазы |
| NIXL | "NVIDIA transport" | inter-node KV transfer Dynamo (RDMA/TCP) |
| NVIDIA Dynamo | "the orchestrator" | stack-above coordinator for vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes native" | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | "Dynamo auto-config" | измеряет workload, конфигурирует pool ratios |
| SLA Planner | "Dynamo policy" | auto-rate-matches prefill:decode to meet SLOs |
| `packDomain: rack` | "llm-d topology" | размещает prefill+decode на одном rack для быстрого KV |
| UCCL | "unified collective" | networking layer llm-d 0.5 для scale-to-zero |
| MoE expert routing | "expert per token" | паттерн DeepSeek-V3; disaggregation помогает |

## Дополнительное чтение

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
