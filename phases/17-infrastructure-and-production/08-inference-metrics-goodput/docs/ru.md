# Метрики инференса — TTFT, TPOT, ITL, Goodput, P99

> Четыре метрики определяют, работает ли deployment инференса. TTFT — это prefill плюс очередь плюс сеть. TPOT (эквивалентно ITL) — ограниченная памятью стоимость decode на токен. Сквозная latency — это TTFT плюс TPOT, умноженный на длину вывода. Throughput — токены в секунду, агрегированные по всему fleet. Но для продукта важен goodput — доля запросов, которые одновременно выполнили все SLO. Высокий throughput при низком goodput означает, что вы обрабатываете токены, которые не доходят до пользователей вовремя. Референсные числа для Llama-3.1-8B-Instruct на TRT-LLM в 2026: mean TTFT 162 ms, mean TPOT 7.33 ms, mean E2E 1,093 ms. Всегда сообщайте P50, P90, P99 — никогда только mean. И следите за ловушкой измерений: GenAI-Perf исключает TTFT из расчета ITL, LLMPerf включает его; два инструмента расходятся по TPOT для одного и того же прогона.

**Тип:** Learn
**Языки:** Python (stdlib, toy percentile calculator and goodput reporter)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals)
**Время:** ~60 minutes

## Цели обучения

- Точно определить TTFT, TPOT, ITL, E2E, throughput и goodput и назвать компонент, который измеряет каждая метрика.
- Объяснить, почему mean — неправильная статистика для LLM serving и как читать P50/P90/P99.
- Сконструировать многокритериальное SLO (например, TTFT<500 ms AND TPOT<15 ms AND E2E<2 s) и вычислить goodput относительно него.
- Назвать два benchmark-инструмента, которые расходятся по TPOT для одного и того же прогона, и объяснить почему.

## Проблема

"Наш throughput — 15,000 tokens per second." И что? Если 40% запросов превысили 2 секунды end-to-end, пользователи бросили сессию. Один throughput не говорит, работает ли продукт.

У инференса несколько осей latency, и каждая ломается по-своему. Prefill ограничен compute и масштабируется с длиной prompt. Decode ограничен памятью и масштабируется с batch size. Задержка в очереди — операционная проблема. Сеть — проблема физического расстояния. Нужны отдельные метрики для каждой оси, нужны перцентили, и нужен единый композитный показатель, который говорит: "получил ли пользователь ожидаемое" — это goodput.

## Концепция

### TTFT — time to first token

`TTFT = queue_time + network_request + prefill_time`

Prefill доминирует при длинных prompts. На Llama-3.3-70B FP8 на H100 prompt 32k занимает ~800 ms чистого prefill. Queue time — поведение scheduler под нагрузкой. Network request — время передачи по сети, включая TLS. TTFT — latency, которую пользователь видит до появления первого streamed token.

### TPOT / ITL — inter-token latency

Много названий для одной величины. `TPOT` (time per output token), `ITL` (inter-token latency), `decode latency per token` — все это одно и то же. Это время между последовательными streamed tokens после первого.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

На том же стеке Llama-3.3-70B H100 с chunked prefill mean TPOT ~7 ms. Без chunked prefill во время длинного prefill соседней последовательности TPOT может подскакивать до 50 ms. Смотрите P99, а не mean.

### E2E latency

`E2E = TTFT + TPOT * output_tokens + network_response`

Для длинных выводов (>500 tokens) E2E определяется TPOT. Для коротких выводов с длинными prompts E2E определяется TTFT. Сообщайте E2E с учетом длины вывода.

### Throughput

`throughput = total_output_tokens / elapsed_time`

Агрегированная метрика. Говорит об эффективности fleet. Не говорит о здоровье отдельных запросов.

### Goodput — метрика, которая действительно важна

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO является многокритериальным. Запрос считается "good" только если выполнены все ограничения. Goodput — эта доля. Высокий throughput при 60% goodput — провал. Меньший throughput при 99% goodput — цель.

В 2026 goodput — метрика, используемая в submissions MLPerf Inference v6.0 и во внутреннем SLA tracking у AI platform providers.

### Почему mean — неправильная статистика

Распределения latency у LLM имеют правый хвост. Decode batch с одним соседним long-prefill может выдать 500 tokens с TPOT ~7 ms и 20 tokens с TPOT ~60 ms. Mean TPOT — 9 ms. P99 TPOT — 65 ms. Пользователи регулярно попадают в P99 — поэтому они уходят.

Всегда сообщайте тройку (P50, P90, P99). Для пользовательского опыта оптимизируют P99.

### Референсные числа — Llama-3.1-8B-Instruct на TRT-LLM, 2026

- mean TTFT: 162 ms
- mean TPOT: 7.33 ms
- mean E2E: 1,093 ms
- P99 TPOT: varies 10-25 ms depending on chunked-prefill configuration.

Это опубликованные референсные точки NVIDIA. Они меняются с размером модели (70B даст 3-5x), hardware (H100 vs B200 ~3x) и нагрузкой.

### Ловушка измерений

Два наиболее используемых в 2026 benchmark-инструмента расходятся по TPOT для одного и того же прогона:

- **NVIDIA GenAI-Perf**: исключает TTFT из расчета ITL. ITL начинается с token 2.
- **LLMPerf**: включает TTFT. ITL начинается с token 1.

Для запроса с TTFT 500 ms и 100 output tokens за 700 ms total decode GenAI-Perf сообщает `ITL = 700/99 = 7.07 ms`, LLMPerf сообщает `ITL = 1200/100 = 12.00 ms`. Выбор инструмента меняет число.

Всегда указывайте инструмент. Всегда публикуйте определение.

### Построение SLO

Разумное consumer-facing SLO для 70B chat model в 2026:

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 s for <300-token outputs.
- Goodput target >= 99%.

Enterprise SLOs ужесточают TTFT (200-400 ms) и ослабляют E2E. Смысл в том, чтобы записать их, измерять все три и отслеживать goodput как единый композит.

### Как измерять

- Запускайте реальный traffic или реалистичный synthetic (LLMPerf с `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Цель — 2x peak concurrency для benchmark run.
- Запускайте 30-50 iterations, берите перцентили объединенной выборки.
- Публикуйте вместе с tool name, tool version, model, hardware, concurrency, prompt distribution.

## Используйте это

`code/main.py` — игрушечный goodput calculator. Он генерирует синтетическое распределение latency, применяет SLO и вычисляет goodput. Также показывает различие TPOT между GenAI-Perf и LLMPerf на одном и том же trace.

## Отправьте в прод

Этот урок создает `outputs/skill-slo-goodput-gate.md`. По workload и SLO он создает CI/CD-ready benchmark recipe, который gate'ит deploys по goodput, а не по throughput.

## Упражнения

1. Запустите `code/main.py`. Сгенерируйте распределение с 1% tail spike. Как изменится goodput, если ужесточить P99 TPOT с 30 ms до 15 ms?
2. Vendor заявляет "15,000 tok/s on Llama 3.3 70B H100". Назовите три вопроса, которые нужно задать, прежде чем доверять этому.
3. Почему chunked prefill защищает P99 TPOT, но не mean TPOT?
4. Сконструируйте consumer SLO для voice assistant (первый token слышат, а не читают). Какая метрика наиболее заметна пользователю?
5. Прочитайте README LLMPerf и docs GenAI-Perf. Найдите еще три метрики, по которым инструменты расходятся.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| TTFT | "time to first token" | Queue + network + prefill; dominated by prefill at long prompts |
| TPOT | "time per output token" | Memory-bound decode cost per token after first |
| ITL | "inter-token latency" | Same as TPOT in most tools (not all — see GenAI-Perf) |
| E2E | "end to end" | TTFT + TPOT * output_len; response-side network on top |
| Throughput | "tok/s" | Fleet efficiency; useless without latency percentiles |
| Goodput | "SLO-met rate" | Fraction of requests meeting every SLO constraint simultaneously |
| P99 | "tail" | 1-in-100 worst-case latency; the user experience metric |
| SLO multi-constraint | "the joint" | AND of all three latency bounds; a request fails if any one is violated |
| GenAI-Perf vs LLMPerf | "the tool trap" | Tools disagree on whether ITL includes TTFT |

## Дополнительное чтение

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — каноническое определение TTFT, ITL, TPOT.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — альтернативные определения и рецепт измерений.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — прикладные измерения на реальных deployments.
- [LLMPerf](https://github.com/ray-project/llmperf) — open-source benchmark на базе Ray.
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — benchmark-инструмент NVIDIA.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — принятый в индустрии benchmark на основе goodput.
