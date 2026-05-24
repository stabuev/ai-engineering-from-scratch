# TensorRT-LLM на Blackwell с FP8 и NVFP4

> TensorRT-LLM работает только на NVIDIA, но выигрывает на Blackwell. На GB200 NVL72 с Dynamo orchestration SemiAnalysis InferenceX измерил $0.012 per million tokens на 120B model в Q1-Q2 2026 против $0.09/M на H100 + vLLM — экономический разрыв 7x. Stack складывает три floating-point regimes: FP8 остается critical для KV cache и attention kernels, потому что у него есть нужный dynamic range; NVFP4 (4-bit microscaling) обслуживает weights и activations; multi-token prediction (MTP) и disaggregated prefill/decode добавляют еще 2-3x сверху. Day-0 model support загружает FP4 weights напрямую без post-training conversion. Загвоздка для engineering teams 2026 года: TRT-LLM — closed NVIDIA stack, поэтому adoption меняет portability на throughput. Посчитайте математику на вашем mix of models and hardware до commitment.

**Тип:** Изучение
**Языки:** Python (stdlib, учебный FP8/NVFP4 memory and cost calculator)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**Время:** ~75 минут

## Цели обучения

- Объяснить, почему FP8 остается critical для KV cache и attention, даже когда weights находятся в NVFP4.
- Посчитать HBM footprint frontier model под BF16, FP8 и NVFP4 и рассуждать, откуда берется экономия.
- Назвать Blackwell-specific features, которые использует TRT-LLM (day-0 FP4, MTP, disaggregated serving, all-to-all primitives).
- Решить, когда NVIDIA-lock в TRT-LLM стоит 7x cost gap относительно vLLM on Hopper.

## Проблема

Frontier inference economics в 2026 году — это "сколько tokens per dollar". Ответ зависит от четырех stacked choices: hardware generation (Hopper H100/H200 vs Blackwell B200/GB200), precision (BF16 → FP8 → NVFP4), serving engine (vLLM vs SGLang vs TRT-LLM) и orchestration (plain vs disaggregated vs Dynamo).

На Hopper с vLLM 120B MoE работает примерно за ~$0.09 per million tokens. На Blackwell с TRT-LLM + Dynamo та же модель работает примерно за ~$0.012 — в 7x дешевле. Часть gap — hardware (Blackwell дает 11-15x per-GPU LLM throughput vs Hopper). Часть — stack: FP4 weights, MTP draft, disaggregated prefill/decode и NVLink 5 all-to-all для MoE expert communication.

Вы не сможете повторить это вне NVIDIA stack. В этом и компромисс: portability for economics. Понять, какие stack choices дают какую долю gap, — цель урока.

## Концепция

### Почему FP8 все еще является floor для KV cache

Распространенная ошибка 2026 года: предполагать, что NVFP4 applies everywhere. Это не так. KV cache требует FP8 (8-bit floating point), потому что он хранит attention keys and values с широким dynamic range. Quantizing KV to FP4 вызывает catastrophic accuracy loss: tail of the distribution drops off и attention scores collapse. Exponent bits FP8 дают KV cache нужный range.

NVFP4 (2025-2026) применяется к weights and activations. Microscaling: каждый block of weights имеет собственный scale factor, поэтому small blocks могут покрывать разные dynamic ranges без per-tensor scale loss. Для activations FP4 держится, потому что activations имеют small-range внутри layer.

Typical Blackwell config:

- Weights: NVFP4 (4-bit microscaling).
- Activations: NVFP4.
- KV cache: FP8.
- Attention accumulator: FP32 (softmax stability).

### Blackwell-specific primitives, которые использует TRT-LLM

- **Day-0 FP4 weights**: model providers ship FP4 weights directly; TRT-LLM loads without post-training conversion. No AWQ / GPTQ step for FP4.
- **Multi-token prediction (MTP)**: та же идея, что EAGLE (Phase 17 · 05), но integrated into the TRT-LLM build.
- **Disaggregated serving**: prefill and decode on separate GPU pools, KV cache transferred over NVLink or InfiniBand. Та же идея, что Dynamo (Phase 17 · 20).
- **All-to-all communication primitives**: NVLink 5 cut MoE expert communication latency by 3x vs Hopper. MoE kernels TRT-LLM tuned for this.
- **NVFP4 + MXFP8 microscaling**: hardware-accelerated scale-factor handling on Blackwell Tensor Cores.

### Числа, которые нужно запомнить

- HGX B200 at $0.02/M tokens on GPT-OSS-120B via TRT-LLM.
- GB200 NVL72 at $0.012/M tokens via Dynamo (orchestrating TRT-LLM).
- H100 + vLLM ≈ $0.09/M tokens on comparable workload.
- 2.8x throughput gain за три месяца обновлений TRT-LLM (2026).
- 11-15x per-GPU LLM throughput, Blackwell vs Hopper.
- MLPerf Inference v6.0 (April 2026): Blackwell dominates every submitted task.

### Сколько качества стоит FP4

NVFP4 агрессивен. На reasoning-heavy workloads (chain-of-thought, math, code-gen with long context) FP4 weights visibly degrade. Per-block calibration mitigates but does not eliminate. Команды, shipping reasoning models, часто используют FP8 weights + FP4 activations как compromise или остаются на H200 with FP8 throughout.

Правило: всегда validate task quality on your eval set before committing to NVFP4 weights.

### Почему это NVIDIA-lock decision

TRT-LLM — C++ + CUDA + closed-source kernels. Models нужно compile for a specific GPU SKU. No AMD, no Intel, no ARM. Если infra strategy multi-vendor, TRT-LLM — non-starter для TRT-LLM-served tier; вы все еще можете serve from vLLM on mixed hardware. Если вы NVIDIA-only, gap 7x оплачивает lock.

### Practical recipe 2026 года

Для annual inference bill $100M+ запуск на Hopper + vLLM оставляет 7-10x on the table. Migrate cost-dominant workloads to Blackwell + TRT-LLM + Dynamo. Keep experimentation tier on H100 + vLLM for model iteration speed. Validate quality on each NVFP4-converted model before production.

### Disaggregation bonus

Disaggregated serving TRT-LLM (separate prefill and decode pools) подробно разбирается в Phase 17 · 20. На Blackwell multiplier stacks: FP4 weights × MTP speedup × disaggregated placement × cache-aware routing. Число 7x предполагает full stack.

## Используйте это

`code/main.py` считает HBM footprint, decode throughput (memory-bound regime) и $/M-tokens для модели в трех stacks: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Запустите его, чтобы увидеть compounding effect и долю gap, которую дает каждое изменение.

## Доведите до результата

Этот урок создает `outputs/skill-trtllm-blackwell-advisor.md`. По workload, model size и annual token volume он решает, стоит ли Blackwell + TRT-LLM stack NVIDIA-lock.

## Упражнения

1. Запустите `code/main.py`. На 120B MoE with 30% active parameters посчитайте memory-bandwidth-limited decode throughput на H100 BF16, H100 FP8 и B200 NVFP4/FP8. Откуда берется самый большой jump?
2. Customer тратит $2M/year на H100 + vLLM. Какое break-even number Blackwell GPUs нужно купить, чтобы amortize migration to TRT-LLM за 12 months, given the 7x economic gap?
3. После NVFP4 weight conversion accuracy на MATH падает на 3 points. Назовите два recovery paths: один quality-first (keep FP8 weights), один cost-first (calibrate with in-domain data).
4. Прочитайте MLPerf v6.0 inference results. У какой task smallest Blackwell-over-Hopper gap и почему?
5. Посчитайте HBM, needed for a 405B model at NVFP4 weights + FP8 KV cache at 128k context. Поместится ли это на single GB200 NVL72 node?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| FP8 | "eight-bit float" | 8-bit floating point; used for KV cache and attention due to dynamic range |
| NVFP4 | "four-bit micro" | NVIDIA's 4-bit microscaling FP format; weights and activations on Blackwell |
| MXFP8 | "MX eight" | Microscaling FP8 variant; hardware-accelerated on Blackwell Tensor Cores |
| Day-0 FP4 | "ship FP4 weights" | Model providers release weights already in FP4; no post-train conversion step |
| MTP | "multi-token prediction" | Integrated speculative-decoding draft TRT-LLM (Phase 17 · 05) |
| Disaggregated serving | "split prefill/decode" | Prefill and decode on separate GPU pools; KV transferred over NVLink/IB |
| All-to-all | "MoE expert comm" | Communication pattern routing tokens to expert GPUs; NVLink 5 cuts 3x |
| InferenceX | "SemiAnalysis inference bench" | Industry-accepted cost-per-token benchmark 2026 года |

## Дополнительное чтение

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — April 2026 MLPerf results.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all and MoE kernels.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — official engine documentation.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — disaggregated orchestration above TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — benchmark suite that publishes Blackwell numbers.
