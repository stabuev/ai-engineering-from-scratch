# GPU Autoscaling на Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

> Три слоя, а не один. Karpenter динамически provisions nodes (менее чем за минуту, на 40% быстрее Cluster Autoscaler). KAI Scheduler обрабатывает gang scheduling, topology awareness и hierarchical queues — он предотвращает ловушку partial allocation 7-of-8, где семь nodes ждут и сжигают деньги из-за одного недостающего GPU. Application-level autoscalers (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) масштабируются по inference-specific signals: queue depth, KV cache utilization, а не CPU/DCGM duty cycle. Классическая ловушка HPA в том, что `DCGM_FI_DEV_GPU_UTIL` — это duty-cycle measurement: 100% может означать 10 requests или 100. vLLM заранее выделяет KV cache memory, поэтому memory никогда не запускает scale-down. Этот урок учит компоновать три слоя и избегать default policy Karpenter `WhenEmptyOrUnderutilized`, которая завершает running GPU jobs прямо во время inference.

**Тип:** Изучение
**Языки:** Python (stdlib, учебный симулятор queue-depth autoscaler)
**Предварительные требования:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Время:** ~75 минут

## Цели обучения

- Нарисовать три слоя autoscaling (node provisioning, gang scheduling, application-level) и назвать инструмент на каждом слое.
- Объяснить, почему `DCGM_FI_DEV_GPU_UTIL` — неправильный HPA signal для vLLM, и назвать две замены (queue depth, KV cache utilization).
- Описать gang scheduling и failure mode partial allocation, который предотвращает KAI Scheduler (7 of 8 GPUs idle).
- Назвать Karpenter consolidation policy (`WhenEmptyOrUnderutilized`), которая завершает running GPU jobs, и сформулировать безопасную альтернативу 2026 года.

## Проблема

Ваша команда запускает LLM-serving service на Kubernetes. Вы настраиваете HPA с `DCGM_FI_DEV_GPU_UTIL` как signal. Сервис упирается в 100% utilization в рабочие часы. HPA не scale up — он уже считает, что вы заполнены. Вы вручную добавляете replica; TTFT падает. HPA все равно не масштабируется. Signal вам врет.

Отдельно вы используете Cluster Autoscaler для nodes. В 2 a.m. приходит 1M-token prompt; cluster тратит 3 минуты на provisioning node, и request timeout.

Еще отдельно вы деплоите 70B model, требующую 8 GPUs на 2 nodes. В cluster свободны 7 GPUs и 1 разбросан по 3 nodes. Cluster Autoscaler provisions node для 1 недостающего GPU. Семь nodes ждут 4 минуты и сжигают деньги, пока Kubernetes поднимает последний GPU.

Три слоя, три разных failure modes. GPU-aware autoscaling в 2026 году — это не "включить HPA". Это композиция node provisioning, gang scheduling и application-signal autoscaling.

## Концепция

### Слой 1 — node provisioning (Karpenter)

Karpenter наблюдает pending pods и provisions nodes примерно за ~45-60 секунд (Cluster Autoscaler обычно занимает 90-120 секунд для GPU nodes). Он динамически выбирает instance types по constraint `NodePool`: если pod требует 8 H100s и в cluster нет подходящего node, Karpenter provisions его напрямую, вместо scaling существующей group.

**Ловушка consolidation**: default `consolidationPolicy: WhenEmptyOrUnderutilized` в Karpenter опасна для GPU pools. Она завершит running GPU node, чтобы мигрировать pods на более дешевый right-sized instance. Для inference workloads это означает evict running requests и заново загрузить 70B model на новом node. Потеря — минуты capacity плюс request failures.

Безопасная настройка для GPU pools:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Позволяет Karpenter consolidate действительно empty nodes через час, но никогда не evict running job.

### Слой 2 — gang scheduling (KAI Scheduler)

KAI Scheduler (project "Karp", затем переименован) обрабатывает то, чего не делает default kube-scheduler:

**Gang scheduling** — schedule all-or-nothing. Distributed inference pod, требующий 8 GPUs, либо запускает все 8 вместе, либо не запускает ни одного. Без этого возникает partial-allocation trap: 7 из 8 pods start, wait indefinitely, burn money.

**Topology awareness** — знать, какие GPUs делят NVLink, какие находятся в одном rack, между какими есть InfiniBand. И размещать pods соответственно. Tensor-parallel workload DeepSeek-V3 67B должен оставаться в одном NVLink domain; KAI Scheduler это учитывает.

**Hierarchical queues** — несколько команд конкурируют за общий GPU pool с priority и quota. Production pinch Team A вытесняется training job Team B только если priority rules разрешают.

KAI деплоится рядом с kube-scheduler как secondary scheduler; workloads аннотируются для его использования. Ray и vLLM production-stack оба интегрируются.

### Слой 3 — application-level signals

**Ловушка HPA**: `DCGM_FI_DEV_GPU_UTIL` — duty-cycle metric: она измеряет, выполнял ли GPU работу на каждом sampling interval. 100% utilization может означать 10 concurrent requests или 100; GPU был busy в обоих случаях. Scaling по duty cycle — это blind scaling.

Хуже того, vLLM и похожие engines pre-allocate KV cache memory (до `--gpu-memory-utilization`). Memory usage остается около 90% даже при одном request. Memory-based HPA никогда не scale down.

**Replacement signals 2026 года**:

- Queue depth (количество requests, ожидающих prefill).
- KV cache utilization (какая доля blocks выделена active sequences).
- Per-replica P99 TTFT (ваш SLA signal).
- Goodput (requests meeting all SLOs per second).

NVIDIA Dynamo Planner и llm-d Workload Variant Autoscaler потребляют эти signals и scale replicas. Они полностью заменяют HPA для LLM serving.

### Когда что использовать

| Решение масштабирования | Инструмент |
|----------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA (or custom HPA on queue depth) |
| Choose GPU type | Karpenter NodePool |
| Preempt low-priority | KAI Scheduler queues |

### Disaggregated prefill/decode все усложняет

Если вы запускаете disaggregated prefill/decode (Phase 17 · 17), у вас два pod classes с разными scaling triggers: prefill pods scale on queue depth, decode pods scale on KV cache pressure. llm-d exposes these as separate `Services` with per-role HPA. Не пытайтесь ставить один HPA перед обоими.

### Cold start тоже важен

Cold-start mitigation (Phase 17 · 10) — место, где node provisioning time становится видимым пользователю. Warm-up Karpenter 45-60 секунд плюс загрузка модели 20GB плюс engine init означает, что from-zero request занимает 2-5 минут. Держите warm pool (`min_workers=1`) для SLO-critical paths или используйте Modal-style checkpointing на application layer.

### Числа, которые нужно помнить

- Karpenter node provisioning: ~45-60s vs Cluster Autoscaler ~90-120s (GPU nodes).
- KAI Scheduler предотвращает partial-allocation waste — 7-of-8 trap.
- `DCGM_FI_DEV_GPU_UTIL` как HPA signal: broken; используйте queue depth или KV utilization.
- Karpenter `WhenEmptyOrUnderutilized`: завершает running GPU jobs. Используйте `WhenEmpty + consolidateAfter: 1h` для inference.

## Используйте это

`code/main.py` симулирует three-layer autoscaler на bursty GPU workload. Сравнивает naive HPA (duty cycle), queue-depth HPA и KAI-gang-scheduled scaling. Выводит unmet requests, idle-GPU minutes и composite score.

## Доведите до результата

Этот урок создает `outputs/skill-gpu-autoscaler-plan.md`. По cluster topology, workload shape и SLO он проектирует three-layer autoscaling plan.

## Упражнения

1. Запустите `code/main.py`. При bursty workload сколько requests теряет naive duty-cycle HPA, которые ловит queue-depth HPA? Откуда возникает разница?
2. Спроектируйте Karpenter NodePool для cluster, обслуживающего Llama 3.3 70B FP8 на H100 SXM5. Укажите `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter` и taint, который не пускает non-GPU workloads на эти nodes.
3. Команда сообщает, что deployments stuck in Pending, потому что "GPUs available but pod won't schedule." Диагностируйте: это Karpenter, kube-scheduler или KAI Scheduler? Какие metrics подтверждают?
4. Выберите signal для autoscale disaggregated prefill pods и другой signal для decode pods. Обоснуйте оба.
5. Посчитайте стоимость ловушки `WhenEmptyOrUnderutilized` consolidation на 24x7 production service, где в среднем 60 request-dropping events/day при P99 TTFT > 10s.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Karpenter | "the node provisioner" | Kubernetes node autoscaler; sub-minute provisioning |
| Cluster Autoscaler | "the old scaler" | Предшественник Kubernetes node autoscaler; slower, group-based |
| KAI Scheduler | "the GPU scheduler" | Secondary scheduler для gang + topology + queues |
| Gang scheduling | "all or nothing" | Schedule N pods atomically или defer all of them |
| Topology awareness | "rack-aware" | Размещение pods по NVLink/IB/rack placement |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU utilization" | Duty-cycle metric; НЕ scaling signal для LLMs |
| Queue depth | "waiting requests" | Correct HPA signal для prefill-bound scaling |
| KV cache utilization | "memory pressure" | Correct HPA signal для decode-bound scaling |
| Consolidation | "Karpenter consolidation" | Node termination на более дешевый instance type |
| `WhenEmpty + 1h` | "safe consolidation" | Policy, которая не evict running GPU jobs |

## Дополнительное чтение

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — design docs and configuration examples.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — semantics consolidation policy and GPU-safe defaults.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner scaling signals.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray integration pattern.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — managed-Kubernetes-specific guidance.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler design.
