# Capstone 14 — Speculative-Decoding Inference Server

> EAGLE-3 в vLLM 0.7 дает 2.5-3x throughput на реальном трафике. P-EAGLE (AWS 2026) продвинул parallel speculation еще дальше. SGLang SpecForge обучал draft heads в масштабе. Red Hat Speculators hub опубликовал aligned drafts для популярных open models. TensorRT-LLM сделал speculative decoding полноценной возможностью на NVIDIA. Production serving stack 2026 года — это vLLM или SGLang с EAGLE-family drafts, FP8 или INT4 quantization и HPA по queue-wait. Задача этого capstone — обслужить две open models с 2.5x+ baseline throughput и полным отчетом по tail-latency.

**Тип:** Capstone
**Языки:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Пререквизиты:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Отрабатываемые фазы:** P3 · P7 · P10 · P17
**Время:** 30 часов

## Цели обучения

- Построить inference-сервер со speculative decoding в стиле EAGLE-3.
- Связать черновик и проверку с управлением KV-кэшем и continuous batching.
- Измерить прирост пропускной способности на реальном трафике.

## Задача

Speculative decoding стал массовой возможностью в 2026 году. EAGLE-3 draft heads обучаются на hidden states target model и предсказывают N tokens вперед; target model проверяет их за один проход. Acceptance rates 60-80% превращаются в 2-3x end-to-end throughput. vLLM 0.7 интегрирует это нативно. SGLang + SpecForge дает training pipeline. Red Hat Speculators публикует aligned drafts для Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B.

Мастерство здесь в serving operations, а не в модели. Acceptance rate дрейфует вместе с traffic distribution (ShareGPT vs code vs domain data). Tail latency при rejection хуже, чем без speculation — нужно отчитаться по p99 на нескольких batch sizes, а не только по steady-state tokens/sec. Cost per 1M tokens vs Anthropic / OpenAI API — ключевой аргумент доверия.

## Концепция

Speculative decoding состоит из двух слоев. **Draft** model (EAGLE-3 head, ngram или меньшая target-aligned model) предлагает k candidate tokens за step. **Target** model проверяет все k за один pass; любой принятый prefix заменяет greedy path. Acceptance rate зависит от draft-target alignment и input distribution.

EAGLE-3 превосходит ngram drafts на большинстве traffic. P-EAGLE запускает parallel speculation для более глубоких draft trees. Trade-off: P99 latency при rejection выше, потому что verify pass крупнее. Serving config должен отчитываться по latency, разбитой по batch-size buckets, чтобы это было видно.

Deployment — Kubernetes. vLLM 0.7 запускает одну replica на GPU или tensor-parallel shard. HPA autoscale работает по queue-wait, а не по CPU. FP8 (Marlin) и INT4 (AWQ) quants удерживают GPU memory в пределах H100 / H200. End-to-end report включает throughput, acceptance rate, p50/p99 при batch 1/8/32 и $/1M tokens.

## Архитектура

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Стек

- Serving: vLLM 0.7 или SGLang 0.4
- Speculative methods: EAGLE-3 draft heads, P-EAGLE parallel speculation, ngram fallback
- Draft training: SpecForge (SGLang) или Red Hat Speculators
- Target models: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- Quantization: FP8 (Marlin), INT4 AWQ
- Deployment: Kubernetes + NVIDIA device plugin; HPA по queue-wait metric
- Eval: ShareGPT, MT-Bench-v2, GSM8K, HumanEval для измерения acceptance по domain spread
- Reference: TensorRT-LLM speculative decoding как vendor baseline

## Сборка

1. **Подготовка target model.** Выбери Llama 3.3 70B. Quantize to FP8 через Marlin. Разверни под vLLM 0.7 на 1xH100 (или 2x tensor-parallel).

2. **Источник draft.** Возьми aligned EAGLE-3 draft head из Red Hat Speculators (или обучи через SpecForge). Загрузи во vLLM speculative-decoding config.

3. **Baseline numbers.** До speculation: tokens/s при batch 1/8/32, p50/p99 latency, GPU utilization. Опубликуй.

4. **Включи EAGLE-3.** Переключи config; повтори тот же benchmark. Отчитай speedup, acceptance rate, p99 tail-latency delta.

5. **P-EAGLE.** Включи parallel speculation; измерь deeper draft tree vs serial EAGLE-3. Отчитай точку перегиба, где P-EAGLE помогает или вредит.

6. **Domain traffic.** Прогони ShareGPT vs HumanEval vs domain-specific traffic через тот же server. Измерь acceptance rate для каждой distribution. Определи, когда drafts дрейфуют.

7. **Вторая target model.** Запусти тот же pipeline на Qwen3-Coder-30B MoE. Draft сложнее (MoE routing noise). Отчитай.

8. **K8s HPA.** Разверни под K8s с HPA, отслеживающим `queue_wait_ms`. Продемонстрируй scale-out при утроении нагрузки.

9. **Сравнение стоимости.** Посчитай $/1M tokens vs Anthropic Claude Sonnet 4.7 and OpenAI GPT-5.4 на том же eval. Опубликуй.

## Использование

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Что сдать

`outputs/skill-inference-server.md` описывает deliverable. Измеренный serving stack со speculative decoding, полным benchmark report и K8s deployment.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Измеренный speedup vs baseline | 2.5x+ throughput при matched quality на двух models |
| 20 | Acceptance rate на реалистичном traffic | Отчет acceptance-rate по каждой distribution |
| 20 | Дисциплина P99 tail-latency | p99 при batch 1/8/32 со speculation и без |
| 20 | Ops | K8s deploy, HPA по queue-wait, плавный rollout |
| 15 | Write-up и методология | Ясное объяснение, что изменилось и почему |
| **100** | | |

## Упражнения

1. Измерь degradation acceptance-rate, когда draft отстает от target на одну версию (например, Llama 3.3 -> 3.4 drift). Построй monitoring alert.

2. Реализуй ngram-fallback: если EAGLE-3 acceptance падает ниже threshold, переключайся на ngram drafts. Отчитай reliability improvement.

3. Проведи контролируемый MoE experiment: тот же Qwen3-Coder-30B с injected routing noise и без него. Измерь sensitivity draft acceptance.

4. Расширь на H200 (141 GB). Отчитай полученный model-size-per-replica headroom и можно ли обслуживать unquantized Llama 3.3 70B.

5. Бенчмаркни TensorRT-LLM speculative decoding на том же H100 hardware. Отчитай, где он выигрывает у vLLM.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|------------------------|
| Draft model | "Speculator" | Маленькая model, предлагающая N tokens для проверки target |
| EAGLE-3 | "2026 draft architecture" | Draft head, обученная на target hidden states; ~75% acceptance |
| P-EAGLE | "Parallel speculation" | Tree of draft branches, проверяемое за один target pass |
| Acceptance rate | "Hit rate" | Доля drafted tokens, принятых без resampling |
| Quantization | "FP8 / INT4" | Lower-precision weights, чтобы уместить больше model в GPU memory |
| Queue wait | "HPA metric" | Время, которое request ждет в pending queue до начала inference |
| Speculators hub | "Aligned drafts" | Red Hat Neural Magic hub с EAGLE drafts для популярных open models |

## Дополнительное чтение

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — эталонный serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — paper по parallel speculative decoding + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — training pipeline для draft-head
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — hub с aligned drafts
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — paper с методом
- [vLLM repository](https://github.com/vllm-project/vllm) — code и benchmarks
