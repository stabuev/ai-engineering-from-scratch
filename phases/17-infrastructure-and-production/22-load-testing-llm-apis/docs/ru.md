# Нагрузочное тестирование LLM APIs — почему k6 и Locust врут

> Традиционные нагрузочные тестеры не проектировались для потоковых ответов, переменной длины вывода, токен-уровневых метрик или насыщения GPU. Большинство команд попадает в две ловушки. Ловушка GIL: токен-уровневые измерения Locust запускают токенизацию под Python GIL, который конкурирует с генерацией запросов при высокой конкурентности; затем очередь токенизации раздувает измеренную межтокенную задержку — узкое место у вас клиент, а не сервер. Ловушка однообразия промптов: одинаковые промпты в цикле тестируют одну точку распределения токенов; реальный трафик имеет переменную длину и разные совпадения префиксов. LLMPerf исправляет это через `--mean-input-tokens` + `--stddev-input-tokens`. Карта инструментов в 2026: LLM-специализированные (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) для токен-уровневой точности; **k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)** — aware к streaming, Kubernetes-native distributed через TestRun/PrivateLoadZone CRDs, лучший вариант для CI/CD gates; Vegeta для Go constant-rate saturation; Locust 2.43.3 только с расширением LLM-Locust для streaming. Паттерны нагрузки: steady-state, ramp, spike (тест autoscaling), soak (утечки памяти).

**Тип:** Build
**Языки:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Предварительные требования:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Время:** ~75 minutes

## Цели обучения

- Объяснить два антипаттерна (GIL trap, prompt-uniformity trap), из-за которых универсальные нагрузочные тестеры врут для LLM APIs.
- Выбрать инструмент под цель: LLMPerf (benchmark run), k6 + streaming extension (CI gate), guidellm (large-scale synthetic), GenAI-Perf (NVIDIA reference).
- Спроектировать четыре паттерна нагрузки (steady, ramp, spike, soak) и назвать failure mode, который ловит каждый.
- Построить реалистичное распределение промптов через mean + stddev входных токенов, а не фиксированную длину.

## Проблема

Вы протестировали LLM endpoint через k6 на 500 concurrent users. Он выдержал. Вы выпустили релиз. В продакшене на 200 реальных пользователях сервис упал — P99 TTFT взлетел, GPUs уперлись в предел.

Произошли две вещи. Во-первых, k6 отправлял 500 одинаковых промптов — request coalescing и prefix caching создали видимость, что вы обрабатываете 500 конкурентных decode, хотя фактически обрабатывали один. Во-вторых, k6 не отслеживает inter-token latency на streaming responses так, как это воспринимает глаз; он видит одно HTTP-соединение, а не 500 токенов, приходящих с разными интервалами.

Нагрузочное тестирование LLM — отдельная дисциплина.

## Концепция

### Ловушка GIL (Locust)

Locust использует Python и выполняет токенизацию на стороне клиента под GIL. При высокой конкурентности tokenizer становится в очередь за генерацией запросов. Измеренная inter-token latency включает backlog клиентской токенизации. Вам кажется, что медленный сервер; на самом деле это тестовый harness.

Исправление: расширение LLM-Locust переносит токенизацию в отдельные процессы, либо используйте harness на компилируемом языке (k6, LLMPerf с tokenizers.rs).

### Ловушка однообразия промптов

Все известные нагрузочные тестеры позволяют настроить один промпт. В циклическом тесте на 10 000 итераций каждый раз отправляется ровно тот же промпт. Сервер каждый раз видит тот же префикс — prefix cache hits приближаются к 100%, throughput выглядит отличным.

Исправление: семплируйте из распределения промптов. LLMPerf использует `--mean-input-tokens 500 --stddev-input-tokens 150` — разные длины, разный контент.

### Четыре паттерна нагрузки

1. **Steady-state** — постоянный RPS 30-60 мин. Ловит: регрессии базовой производительности.
2. **Ramp** — линейное увеличение RPS от 0 до целевого за 15 мин. Ловит: capacity breakpoint, аномалии прогрева.
3. **Spike** — резкий 3-10x RPS на 2 мин, затем возврат. Ловит: задержку autoscaling, saturation очередей, влияние cold start.
4. **Soak** — steady-state 4-8 часов. Ловит: утечки памяти, дрейф connection pools, переполнение observability.

### Карта инструментов 2026

**LLMPerf** (Anyscale) — Python, но токенизация на Rust. Mean/stddev промпты. Streaming-aware. Лучший дефолт для performance runs.

**NVIDIA GenAI-Perf** — референс NVIDIA. Использует Triton client; широкий охват метрик. Учтите: его ITL исключает TTFT; у LLMPerf включает. Два инструмента дают разный TPOT для одного сервера.

**LLM-Locust** (TrueFoundry) — расширение Locust, исправляющее GIL trap. Привычный Locust DSL + streaming metrics.

**guidellm** — large-scale synthetic benchmarking.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)**:
- сам k6 (Go, compiled, без GIL) добавил streaming-aware metrics.
- k6 Operator использует TestRun / PrivateLoadZone CRDs для Kubernetes-native distributed testing.
- Лучший вариант для CI/CD gates и SLA testing.

**Vegeta** — Go, проще k6. Constant-rate HTTP saturation. Не LLM-aware, но хорош для тестирования gateway / rate limit.

**Locust 2.43.3 stock** — имеет GIL trap для LLM. Только с расширением LLM-Locust.

### SLA gate в CI

Запускайте k6 на PR с:

- 30-50 итерациями на baseline RPS.
- Gate: P50/P95 TTFT, 5xx < 5%, TPOT ниже порога.
- Ломайте build при нарушении.

### Реалистичное распределение промптов

Стройте его из реальных сэмплов трафика (если они есть) или опубликованных распределений (например, ShareGPT prompts для chat, HumanEval для code). Передайте mean + stddev в LLMPerf. Избегайте loop-with-one-prompt любой ценой.

### Числа, которые стоит запомнить

- k6 Operator 1.0 GA: September 2025.
- k6 v2026.1.0: streaming-aware metrics.
- Типичный LLMPerf run: 100-1000 requests при concurrency X.
- Типичный CI gate: 30-50 iterations на PR.
- Четыре паттерна: steady, ramp, spike, soak.

## Используйте это

`code/main.py` симулирует нагрузочный тест с реалистичным распределением промптов, измеряет effective TPOT и демонстрирует ловушку uniform-prompt.

## Отгрузите это

Этот урок создает `outputs/skill-load-test-plan.md`. По workload и SLA выбирает инструмент и проектирует четыре паттерна нагрузки.

## Упражнения

1. Запустите `code/main.py`. Сравните uniform vs realistic distribution — где разрыв?
2. Напишите k6 script для CI gate: TTFT P95 < 800 ms при 100 concurrent, runtime 5 minutes.
3. Soak test показывает рост памяти 50 MB/hour. Назовите три причины и instrumentation, чтобы выбрать между ними.
4. Spike test с 10 RPS до 100 RPS. Какое expected recovery time, если Karpenter + vLLM production-stack уже стоят (Phase 17 · 03 + 18)?
5. GenAI-Perf сообщает TPOT=6ms; LLMPerf сообщает TPOT=11ms на том же сервере. Объясните.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| LLMPerf | "the LLM harness" | Benchmark-инструмент Anyscale, streaming-aware |
| GenAI-Perf | "NVIDIA tool" | Референсный harness NVIDIA |
| LLM-Locust | "Locust for LLMs" | Расширение Locust, исправляющее GIL trap |
| guidellm | "synthetic benchmark" | Large-scale synthetic tool |
| k6 Operator | "K8s k6" | Distributed k6 на базе CRD |
| GIL trap | "Python client overhead" | Backlog токенизации раздувает измеренную latency |
| Prompt-uniformity trap | "single-prompt lie" | Цикл с тем же промптом попадает в cache и раздувает throughput |
| Steady-state | "constant load" | Ровный RPS N минут |
| Ramp | "linear up" | От 0 до target за duration |
| Spike | "burst test" | Резкий multiplier, затем возврат |
| Soak | "long test" | Часы для обнаружения утечек |

## Дополнительное чтение

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
