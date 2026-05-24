# Multi-Region LLM Serving and KV Cache Locality

> Round-robin load balancing активно вреден для cached LLM inference. Запрос, который не попал на node с его prefix, платит full prefill cost — примерно 800 ms at P50 на long prompt против ~80 ms при cache hit. В 2026 production pattern — cache-aware router (vLLM Router in Rust, llm-d router), который потребляет KV-cache events и маршрутизирует по prefix-hash match. Новые исследования (GORGO) делают cross-region network latency явным членом routing objective. Коммерческие предложения "cross-region inference" (Bedrock cross-region inference, GKE multi-cluster gateways) рассматривают inference как opaque — они решают availability, а не TTFT. JPMorgan и Mayo Clinic провели us-east-1 failover в Nov 2024 примерно за 22 minutes. Реальность DR: 32% LLM DR failures происходят потому, что команды сделали backup weights, но забыли tokenizer files или quantization configs.

**Тип:** Learn
**Языки:** Python (stdlib, toy prefix-cache-aware router simulator)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Время:** ~60 minutes

## Цели обучения

- Объяснить, почему round-robin load balancing ломает cached inference, и количественно оценить TTFT penalty.
- Нарисовать cache-aware router: inputs (KV-cache events), algorithm (prefix-hash match), tie-breaker (GPU utilization).
- Назвать 32% DR failure driver для LLMs (missing tokenizer files / quantization configs) и сформулировать three-file DR checklist.
- Отличить commercial cross-region offerings (Bedrock CRI, GKE Multi-Cluster Gateway) от KV-aware routing.

## Проблема

Ваш сервис работает в us-east-1, us-west-2 и eu-west-1. Вы ставите ALB перед ним с round-robin. Prefix cache hit rate в production падает до 8%. TTFT P50 утраивается. Логи vLLM показывают, что каждый request платит full prefill cost.

Round-robin оптимален для stateless services. LLM inference по своей природе stateful — KV cache кодирует все, что model уже видела. Blind routing — это routing в неправильный cache.

Отдельно у вашей команды есть DR plan. Вы делаете backup model weights в S3 cross-region. Случается regional outage; вы пытаетесь failover; replica отказывается стартовать. Вы забыли, что tokenizer.json, quantization config и RoPE scaling config были в отдельном bucket, который вы не sync.

Multi-region LLM serving — это cache problem, routing problem и DR-hygiene problem, а не load-balancer problem.

## Концепция

### Cache-aware routing

Приходит request с prompt. Router хэширует prefix (например, first 512 tokens); он спрашивает каждую replica: "есть ли у тебя этот prefix cached?" Replicas публикуют KV-cache events в pub/sub channel по мере allocate и evict blocks. Router выбирает replica с match, а если match нет, fall through к tie-breaker на основе GPU-util.

**vLLM Router** (Rust, 2026 production-stack): подписывается на `kv.cache.block_added` events, поддерживает prefix-hash → replica index, маршрутизирует через O(1) lookup. Falls through к least-queue-depth, когда match нет.

**llm-d router**: тот же pattern, Kubernetes-native. Публикует events через ControlPlane API.

**SGLang RadixAttention** (Phase 17 · 06) — intra-replica equivalent. Cross-replica routing строго upstream.

### Числа

TTFT P50 на 2K-token prompt, Llama 3.3 70B FP8, H100:
- Cache hit (same replica, prefix resident): ~80 ms.
- Cache miss (cold prefill): ~800 ms.

Разрыв 10x. Если router достигает 60-80% prefix cache hits across replicas, вы приближаетесь к single-replica performance при N-replica capacity. Если hits 10%, вы приближаетесь к naive scaling.

### Cross-region получает новое ограничение — network latency

Inter-region RTT:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

Если routing отправляет request из us-east-1 к hot prefix в ap-southeast-1, saved prefill (800 → 80 ms) перекрывается 440 ms round-trip. GORGO (2026 research) делает это явным — минимизируйте `prefill_time + network_latency` совместно, а не только prefill. Часто ответ — держать routing regional, кроме massive multi-MB prefixes, где prefill доминирует.

### Commercial "cross-region inference" здесь не помогает

AWS Bedrock cross-region inference автоматически routes requests в другие regions при capacity pressure. Он оптимизирует availability, не TTFT, и treats inference as opaque. GKE Multi-Cluster Gateway делает то же самое — service-level failover без awareness of KV cache.

Вам все равно нужен app-layer cache-aware router даже при использовании этих систем. Они решают случай "us-east-1 is on fire". Cache-aware routing решает случай TTFT.

### DR hygiene — проблема 32% missing-files

Широко цитируемая статистика 2026: 32% LLM DR failures происходят потому, что команды сделали backup weights, но забыли:

- `tokenizer.json` or `tokenizer.model`
- Quantization configs (`quantize_config.json`, AWQ scales, GPTQ zero-points)
- Model-specific configs (RoPE scaling, attention masks, chat templates)
- Engine config (`vllm_config.yaml`, sampling defaults, LoRA adapter manifests)

Исправление — three-file minimum DR manifest:

1. All files under the HF model repo (weights + configs + tokenizer).
2. Engine-specific serving config.
3. Deployment manifest (K8s YAML, Dockerfile, dependency lock).

Плюс: проводите DR drill quarterly. JPMorgan us-east-1 drill достиг 22 minutes recovery in Nov 2024 только потому, что playbook был отрепетирован.

### Data residency ортогональна

EU customer PHI не может покидать EU. Если cache-aware router отправляет Paris-originated request в us-east-1 ради prefix match, вы нарушили GDPR независимо от выигрыша TTFT. Разделяйте routers по residency boundary до оптимизации cache.

### Числа, которые нужно помнить

- Cache hit vs miss TTFT gap: ~10x (80 ms vs 800 ms on 2K prompt).
- Inter-region RTT US-EU: ~75 ms.
- DR failure: 32% miss tokenizer/quant configs.
- JPMorgan us-east-1 failover Nov 2024: 22 minutes (30-min SLA).

## Используйте это

`code/main.py` симулирует три routing strategies (round-robin, cache-aware regional, cache-aware global) на multi-region workload. Сообщает cache hit rate, TTFT P50/P99 и cross-region bill.

## Отправьте в прод

Этот урок создает `outputs/skill-multi-region-router.md`. По regions, residency constraints и SLA он проектирует routing plan.

## Упражнения

1. Запустите `code/main.py`. При какой prompt length cross-region routing превосходит local-only routing, given 75 ms RTT?
2. Ваш cache hit rate падает с 70% до 12%. Диагностируйте три возможные причины и observables, которые подтвердят каждую.
3. Спроектируйте DR manifest для 70B AWQ-quantized model served in vLLM with 5 LoRA adapters. Перечислите каждый file and config.
4. Аргументируйте, достаточно ли Bedrock cross-region inference для fintech со strict TTFT SLOs. Сошлитесь на конкретное behavior.
5. Paris-origin request matches a prefix in us-east-1. Вы routing it? Напишите policy.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Cache-aware routing | "smart LB" | Route on prefix-hash match to KV-cache-holding replica |
| KV-cache events | "cache pub-sub" | Replicas publish block add/evict; router indexes |
| Prefix hash | "cache key" | Hash of first N tokens used as router lookup |
| GORGO | "cross-region routing research" | arXiv 2602.11688; network latency as explicit term |
| Cross-region inference | "Bedrock CRI" | AWS product; availability failover, not TTFT awareness |
| DR manifest | "the backup list" | Every file needed to restore — not just weights |
| Data residency | "GDPR boundary" | Legal constraint on which region sees user data |
| RTT | "round-trip time" | Network latency; 75 ms US-EU, 220 ms US-APAC |
| LLM-aware LB | "cache-hit LB" | Cache-aware router as a product category |

## Дополнительное чтение

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — cross-region KV-cache reuse with network latency term.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — availability failover documentation.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — source cache-aware router.
