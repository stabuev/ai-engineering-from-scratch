# vLLM Production Stack with LMCache KV Offloading

> vLLM's production-stack — reference Kubernetes deployment: router, engines и observability связаны вместе. LMCache — слой KV-offloading, который выносит KV cache из GPU memory и переиспользует его между queries и engines (CPU DRAM, затем disk/Ceph). vLLM 0.11.0 KV Offloading Connector (January 2026) делает это асинхронным и pluggable через Connector API (v0.9.0+). Offload latency не видна пользователю напрямую. LMCache полезен даже без shared prefixes — когда у GPU заканчиваются KV slots, preempted requests можно восстановить из CPU вместо повторного вычисления prefill. Published benchmarks на 16x H100 (80GB HBM) across 4 a3-highgpu-4g: когда KV cache превышает HBM, и native CPU offload, и LMCache существенно повышают throughput; при низком KV footprint все configs совпадают с baseline с небольшим overhead.

**Тип:** Learn
**Языки:** Python (stdlib, toy KV-spill simulator)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Время:** ~60 minutes

## Цели обучения

- Нарисовать слои vLLM production-stack: router, engines, KV offload, observability.
- Объяснить KV Offloading Connector API (v0.9.0+) и то, как asynchronous path в 0.11.0 скрывает offload latency.
- Количественно определить, когда LMCache CPU-DRAM помогает (KV > HBM), а когда добавляет overhead (KV достаточно мал, чтобы поместиться в HBM).
- Выбрать между native vLLM CPU offload и LMCache connector с учетом deployment constraints.

## Проблема

Ваш vLLM serving показывает GPUs at 100% HBM и preemption events при росте concurrency. Requests evicted, requeued, и вы повторно выполняете prefill одного и того же 2K-token prompt четыре раза в минуту. GPU compute тратится на redundant prefills; goodput заметно ниже raw throughput.

Добавление GPUs стоит линейно. Добавить HBM невозможно. Но CPU DRAM дешевая — один socket имеет 512 GB+ с latency на порядки хуже HBM, но достаточно хорошей для "temporarily warm" KV cache.

LMCache выносит KV cache в CPU DRAM, чтобы preempted requests быстро восстанавливались, а repeated prefixes между engines переиспользовали cache без повторного prefill на каждом engine.

## Концепция

### vLLM production-stack

`github.com/vllm-project/production-stack` — reference Kubernetes deployment:

- **Router** — cache-aware (Phase 17 · 11). Потребляет KV events.
- **Engines** — vLLM workers. Один на GPU или на TP/PP group.
- **KV cache offload** — LMCache deployment или native connector.
- **Observability** — Prometheus scrape, Grafana dashboards, OTel traces.
- **Control plane** — service discovery, config, rolling updates.

Поставляется как Helm chart + operator.

### The KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 ввел Connector API для pluggable KV cache backends. Engine offloads blocks в connector; connector хранит их (RAM, disk, object storage, LMCache). Когда request нужен block, connector загружает его обратно.

vLLM 0.11.0 (January 2026) добавляет asynchronous offload path — offload может происходить в фоне, поэтому engine в обычном случае на нем не блокируется. End-to-end latency и throughput все равно зависят от workload shape, KV cache hit rate и system pressure; собственные notes vLLM отмечают, что custom-kernel offload может ухудшать throughput при low hit rates, а async scheduling имеет известные interaction issues со speculative decoding.

### Native CPU offload vs LMCache

**Native vLLM CPU offload**: engine-local. Хранит KV blocks в host RAM. Быстро внедряется, zero network hop. Не работает между engines.

**LMCache connector**: cluster-scale. Хранит blocks в shared LMCache server (CPU DRAM + Ceph/S3 tier). Blocks доступны любому engine. Опубликованы benchmarks на 16x H100.

Выбирайте native, когда HBM pressure есть у одного engine. Выбирайте LMCache, когда несколько engines делят prefixes (RAG with common system prompts, multi-tenant with shared templates).

### Benchmark behavior

Тест 16x H100 (80 GB HBM), распределенных по 4 a3-highgpu-4g:

- Low KV footprint (short prompts, low concurrency): все configs совпадают с baseline, LMCache добавляет ~3-5% overhead.
- Moderate footprint: LMCache начинает помогать на prefix reuse между engines.
- KV exceeds HBM: native CPU offload и LMCache оба существенно повышают throughput; LMCache дает больший gain из-за cross-engine sharing.

### When LMCache is decisive

- Multi-tenant serving, где system prompts общие между tenants.
- RAG, где document chunks повторяются между queries.
- Fine-tuned variants (LoRA) на одной base, где base-model KV reuse сокращает redundant work.
- Preemption-heavy workloads: restore from CPU дешевле, чем re-prefill.

### When NOT to enable

- Small HBM pressure — вы платите overhead без пользы.
- Short contexts (<1K tokens) — transfer time > re-prefill.
- Single-tenant single-prompt workload — нет reuse, который можно забрать.

### Integration with disaggregated serving

Phase 17 · 17 disaggregated serving + LMCache усиливают друг друга: KV transfers from prefill pool to decode pool попадают в LMCache, если не используются; последующие queries забирают из LMCache. Phase 17 · 11 cache-aware router может направить request к engine, где совпадает local OR LMCache-shared cache.

### Numbers you should remember

- vLLM 0.9.0: Connector API shipped.
- vLLM 0.11.0 (Jan 2026): asynchronous offload path; end-to-end latency impact depends on workload, KV hit rate, and system pressure (not an absolute guarantee).
- 16x H100 benchmark: LMCache helps when KV footprint exceeds HBM.
- Small HBM pressure: 3-5% overhead without benefit.

## Используйте это

`code/main.py` симулирует preemption-heavy workload с LMCache и без него. Показывает avoided re-prefills, throughput gain и break-even HBM utilization.

## Доведите до результата

Этот урок создает `outputs/skill-vllm-stack-decider.md`. По workload shape и vLLM deployment он выбирает native vs LMCache vs neither.

## Упражнения

1. Запустите `code/main.py`. При какой HBM utilization LMCache начинает окупаться?
2. Tenant использует общий 6K-token system prompt для 200 queries/hour. Посчитайте expected LMCache savings per tenant.
3. LMCache server — single point of failure. Спроектируйте HA strategy (replicas, fallback to native).
4. LMCache stores to Ceph on spinning disk. For a 4K-token KV at 70B FP8 (500 MB), what's the read time vs re-prefill?
5. Аргументируйте, является ли vLLM 0.11.0 asynchronous path "free" — где скрывается overhead?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Production-stack | "the reference deployment" | vLLM's Kubernetes Helm chart + operator |
| Connector API | "KV backend interface" | pluggable KV store interface vLLM 0.9.0+ |
| Native CPU offload | "engine-local spill" | хранение KV в host RAM того же engine |
| LMCache | "cluster KV cache" | cross-engine KV cache server on CPU DRAM + disk |
| 0.11.0 async | "non-blocking offload" | offload, скрытый за engine stream |
| Preemption | "evict to make room" | KV cache shuffle при заполненном HBM |
| Prefix reuse | "same system prompt" | несколько queries имеют общее начало; cache hit |
| Ceph tier | "disk tier" | durable storage ниже DRAM в cache hierarchy |

## Дополнительное чтение

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details.
