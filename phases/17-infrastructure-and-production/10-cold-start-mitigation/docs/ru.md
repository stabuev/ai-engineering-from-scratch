# Cold Start Mitigation for Serverless LLMs

> Образ модели на 20 GB занимает от 5-10 минут (7B) до 20+ минут (70B), чтобы пройти путь от cold до serving. В настоящем serverless мире это не warm-up — это outage. Mitigations работают на пяти слоях: pre-seeded node images (Bottlerocket on AWS, dual-volume arch), model streaming (NVIDIA Run:ai Model Streamer, native in vLLM), GPU memory snapshots (Modal checkpoints, до 10x faster restart), warm pools (`min_workers=1`), tiered loading (ServerlessLLM's NVMe→DRAM→HBM pipeline, 10-200x latency reduction) и live migration, которая переносит input tokens (KB), а не KV cache (GB). Modal публикует 2-4s cold starts как нижнюю границу; Baseten 5-10s default, sub-second with pre-warming. Этот урок учит измерять, бюджетировать и складывать пять слоев.

**Тип:** Learn
**Языки:** Python (stdlib, toy cold-start path simulator)
**Предварительные требования:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Время:** ~60 minutes

## Цели обучения

- Перечислить пять слоев cold-start mitigation и назвать один tool или pattern на каждом слое.
- Вычислить total cold-start time как сумму (node provision) + (weights download) + (weights load into HBM) + (engine init) для 70B model.
- Объяснить, почему live migration передает input tokens (KB), а не KV cache (GB), и какова цена (recomputation).
- Назвать trade-off warm-pool (платить за idle GPU или принимать cold-start tail) и SLA threshold, при котором `min_workers > 0` становится обязательным.

## Проблема

Ваш serverless LLM endpoint ночью scale to zero. В 8 a.m. traffic spikes. Первый запрос ждет, пока:

1. Karpenter provisions a GPU node: 45-60s.
2. The container pulls a 30 GB image with weights: 120-300s.
3. The engine loads weights into HBM: 45-120s depending on model size and storage speed.
4. vLLM or TRT-LLM initializes CUDA graphs, KV cache pool, tokenizer: 10-30s.

Итого: 220-510s (примерно 3-8 минут) до первого token. Ваш SLA — 2s. Вы ship warm-pool (`min_workers=1`), и проблема как будто исчезает — но теперь вы платите за один idle GPU 24x7. Если у сервиса 5 products, каждый с одной warm replica, это 5 × 24 × 30 = 3,600 GPU-hours/month, независимо от того, вызвал ли хоть один пользователь сервис.

Cold-start mitigation — это способ сохранить serverless economics, приближаясь к latency always-on.

## Концепция

### Layer 1 — pre-seeded node images (Bottlerocket)

На AWS dual-volume architecture Bottlerocket отделяет OS от data. Сделайте snapshot data volume с уже pulled container image; укажите snapshot ID в `EC2NodeClass`. Новые nodes загружаются с weights уже на local NVMe — steps 2 и часть 3 исчезают. Нативно работает с Karpenter. Типичная экономия: 2-4 minutes на cold start для large models.

Эквивалент на GCP: custom VM images с заранее baked container layers. На Azure: managed disk snapshots по тому же pattern.

### Layer 2 — model streaming (Run:ai Model Streamer)

Вместо загрузки всего файла до ответа на первый request, stream weights в GPU memory layer-by-layer и начинайте processing, как только первый transformer block resident. NVIDIA Run:ai Model Streamer поставляется native in vLLM 2026. Работает с S3, GCS и local NVMe. Сокращает weight-load time примерно вдвое для large models за счет overlap I/O with compute setup.

### Layer 3 — GPU memory snapshots (Modal)

Modal делает checkpoint GPU state (weights, CUDA graphs, KV cache region) после первой загрузки. Последующие restarts deserialize напрямую в HBM — 10x быстрее, чем re-initializing. Это ближайшая вещь к "boot a warm GPU in 2 seconds." Trade-off: snapshots зависят от per-GPU-topology, поэтому если Karpenter переносит вас на другой SKU, нужно re-checkpoint.

### Layer 4 — warm pools (min_workers=1)

Самая простая mitigation: держать одну replica всегда ready. Стоимость — hourly rate одного GPU 24x7. Арифметика жестока на small models (вы платите $0.85-$1.50/hr, чтобы избежать 30s cold start) и благосклонна к large ones (платите $4/hr, чтобы избежать 5-minute cold start). SLA threshold, где warm pools становятся обязательными: обычно TTFT P99 < 60s на 70B+ model.

### Layer 5 — tiered loading (ServerlessLLM)

ServerlessLLM рассматривает storage как hierarchy: NVMe (fast but big), DRAM (medium but tiered), HBM (tiny but instant). Weights предварительно загружаются в DRAM; load-on-demand в HBM. Paper сообщает 10-200x latency reduction на cold loads относительно naive disk-to-HBM. Production adoption ранний, но integrations with vLLM существуют.

### Layer 6 — live migration (bonus pattern)

Когда node становится недоступной (spot eviction, node drain), традиционный pattern — cold-start другой replica и drain request queue. Live migration переносит input tokens (kilobytes) на destination, где model loaded, и recomputes KV cache на destination. Recomputation дешевле, чем передавать GB KV cache по сети. Применимо к disaggregated deployments.

### Математика warm-pool

Для сервиса с P99 TTFT SLA of 2s вопрос не "warm pool yes/no", а "сколько warm replicas и какие paths их получают".

- High-value interactive paths (live chat, voice agent): `min_workers=1-2`.
- Background batch paths (nightly classification): scale-to-zero accepted, 5-10 minute cold start tolerable.
- Premium tier: `min_workers` per tenant with dedicated capacity.

### Измеряйте перед оптимизацией

Анатомия cold-start для 70B model на fresh node (illustrative):

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total with mitigations** | **~15s** | 22x reduction |

### Числа, которые нужно помнить

- Modal cold start: 2-4s (with GPU snapshots).
- Baseten default cold start: 5-10s; sub-second with pre-warming.
- Raw 70B cold start: 3-8 minutes.
- Run:ai Model Streamer: ~2x weight-load speedup.
- ServerlessLLM tiered loading: 10-200x latency reduction (paper numbers).

## Используйте это

`code/main.py` моделирует cold-start path с каждой mitigation и без нее. Сообщает total cold-start time, warm-pool cost и break-even request rate, выше которого warm pool окупается.

## Отправьте в прод

Этот урок создает `outputs/skill-cold-start-planner.md`. По SLA, model size и traffic shape выбирает, какие mitigations складывать.

## Упражнения

1. Запустите `code/main.py`. Вычислите break-even request rate, выше которого warm replica дешевле, чем платить cold-start tax через extra request drops at SLO.
2. Вы deploy 13B model с P99 TTFT SLA of 3s. Выберите минимальный mitigation stack (fewest layers), который этого достигает.
3. Bottlerocket pre-seeding устраняет image pull, но weights все еще загружаются из snapshot в HBM. Вычислите wall-clock для 70B model, если snapshot-backed NVMe читает at 7 GB/s.
4. Ваш serverless provider предлагает GPU snapshots (Modal), а команда отказывается, потому что "snapshots leak PII." Аргументируйте обе стороны — каков realistic risk и mitigation (ephemeral snapshots, encryption, namespace isolation)?
5. Спроектируйте tiered warm-pool policy: сколько warm replicas для paid users, trial users и batch workloads? Покажите математику.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Cold start | "the big pause" | Time from request to first token on a fresh replica |
| Warm pool | "always-on minimum" | `min_workers >= 1` to keep at least one replica ready |
| Pre-seeded image | "baked AMI" | Node image with container weights pre-resident |
| Bottlerocket | "AWS node OS" | AWS container-optimized OS with dual-volume snapshot support |
| Model streamer | "streaming load" | Overlap weights I/O with compute setup |
| GPU snapshot | "checkpoint to HBM" | Serialize post-load GPU state; deserialize on restart |
| Tiered loading | "NVMe + DRAM + HBM" | Hierarchy of storage tiers; load on demand |
| Live migration | "move tokens" | Transfer input (KB), recompute KV on destination |
| `min_workers` | "warm replicas" | Serverless minimum keep-alive count |
| Scale-to-zero | "full serverless" | No cost when idle; accept full cold-start tax |

## Дополнительное чтение

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — опубликованные Modal benchmarks и checkpoint architecture.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot pattern.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — overlap weights load with compute setup.
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — playbook pre-warming.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — design tiered loading.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — live migration для disaggregated deployments.
