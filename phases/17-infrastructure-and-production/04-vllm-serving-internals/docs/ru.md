# Внутреннее устройство vLLM Serving: PagedAttention, Continuous Batching, Chunked Prefill

> Доминирование vLLM в 2026 году держится на трех усиливающих друг друга defaults, а не на одном трюке. PagedAttention всегда включен. Continuous batching вставляет новые requests в active batch между decode iterations. Chunked prefill режет длинные prompts так, чтобы decode tokens никогда не голодали. Включите все три, и Llama 3.3 70B FP8 на одном H100 SXM5 выдает 2,200-2,400 tok/s при 128 concurrent — примерно на 25% выше собственного default vLLM и в 3-4x выше naive PyTorch loop. Этот урок разбирает scheduler и attention kernel на уровне, который можно нарисовать, и заканчивается учебным continuous batcher в `code/main.py`, который schedules prefill и decode так же, как vLLM.

**Тип:** Изучение
**Языки:** Python (stdlib, учебный continuous batching scheduler)
**Предварительные требования:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Время:** ~75 минут

## Цели обучения

- Объяснить PagedAttention как KV cache allocator: blocks, block tables и почему fragmentation остается ниже 4% при production load.
- Нарисовать continuous batching на уровне iteration: как finished sequences выходят из batch, а новые входят без draining.
- Описать chunked prefill одним предложением и назвать, какую latency metric он защищает (подсказка: это TTFT tail, а не mean throughput).
- Назвать gotcha vLLM v0.18.0 2026 года, которая кусает команды, включающие все optimizations сразу.

## Проблема

Naive PyTorch serve loop запускает один request за раз: tokenize, prefill, decode until EOS, return. Для одного пользователя это работает. Для ста — это очередь терпеливых людей. Очевидное исправление — static batching — pads every request to the longest prompt in the window, pads every decode to the longest expected output, и тормозит весь batch на самой медленной sequence. Вы платите за padding, который никогда не используете, а быстрые requests ждут медленные.

vLLM решает три проблемы сразу. PagedAttention не дает KV cache fragmentation съесть 60-80% GPU memory, как это делает классическая contiguous allocation. Continuous batching позволяет requests входить и выходить из batch между каждым decode iteration, поэтому batch всегда заполнен реальной работой. Chunked prefill разбивает 32k-token prompt на slices примерно по 512 tokens, которые interleave with decode, чтобы длинный prompt не замораживал каждый decode token на GPU.

Production default 2026 года — все три включены. Нужно понимать, что делает каждый, потому что failure modes находятся в scheduler, а не в модели.

## Концепция

### PagedAttention как система virtual memory

KV cache равен `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` на sequence. Для Llama 3.3 70B при 8192 tokens это примерно 1.25 GB на sequence в BF16. Если заранее reserve 8192 slots для каждого request, но average request использует только 1500 tokens, вы waste примерно 82% зарезервированной HBM. Classic batching платит за этот waste.

PagedAttention заимствует идею из OS virtual memory. KV cache не является contiguous per sequence. Он выделяется fixed-size blocks (default 16 tokens). У каждой sequence есть block table, которая maps logical token positions to physical block IDs. Когда sequence растет дальше выделенных blocks, добавляется еще один block. Когда она завершается, blocks возвращаются в pool.

Fragmentation падает с 60-80% (classic) до ниже 4% (PagedAttention). Вы не включаете PagedAttention флагом — это единственный allocator, который поставляет vLLM. Knob — `--gpu-memory-utilization` (default 0.9), который говорит vLLM, сколько HBM reserve for KV blocks после загрузки weights and activations.

### Continuous batching на уровне iteration

Старый "dynamic batching" ждал window (например, 10 ms), чтобы заполнить batch, затем выполнял prefill + decode + decode + decode, пока каждая sequence не завершится. Fast sequences уходили рано и простаивали, пока GPU заканчивал медленные.

Continuous batching работает между каждым decode step. Назовем set of running sequences списком `RUNNING`. На каждой iteration:

1. Любая sequence в `RUNNING`, которая только что достигла EOS или max_tokens, удаляется.
2. Scheduler смотрит на waiting queue. Если есть свободные KV blocks, он admits new sequences (prefill или resumed).
3. Forward pass запускается на том, что теперь находится в `RUNNING`, выдавая один новый token per sequence.

Batch size никогда не pads to fixed number. Sequences на разных positions in their output делят один fused forward. В vLLM 2026 года это называется `V1 scheduler`. Ключевой invariant: scheduler runs once per decode iteration, not once per request.

### Chunked prefill защищает TTFT tail

Prefill compute-bound. 32k-token prompt на Llama 3.3 70B занимает ~800 ms чистого prefill на одном H100. Пока prefill runs, decode tokens для всех остальных sequences in the batch ждут. В serving loop first-token latency (TTFT) одного длинного prompt становится inter-token latency (ITL) blip для десятков других пользователей.

Chunked prefill splits prefill into fixed-size chunks (default 512 tokens) и schedules each chunk as a unit. Между chunks scheduler может продвинуть decode sequences на один token. Вы обмениваете небольшой absolute prefill latency hit (несколько ms на chunk) на гораздо меньший decode-time jitter. P99 ITL under mixed load падает с ~50 ms до ~15 ms в опубликованных benchmarks.

### Три defaults взаимодействуют

Все три features предполагают друг друга. PagedAttention дает scheduler fine-grained KV resource, которым можно торговать. Continuous batching нуждается в этом fine-grained resource, чтобы admitting a new sequence не forced global reshuffle. Chunked prefill — это решение scheduler на том же списке `RUNNING`; это еще одна scheduler policy, а не отдельная система.

Не нужно знать каждый flag. Нужно знать, что оптимизирует scheduler: goodput under KV-block budget, subject to chunked prefill slicing.

### Gotcha v0.18.0 в 2026 году

В vLLM v0.18.0 нельзя сочетать `--enable-chunked-prefill` с draft-model speculative decoding (`--speculative-model`). Документированное исключение — N-gram GPU speculative decoding в V1 scheduler. Команды, которые включают каждый flag без чтения release notes, получают run-time error at startup, а не soft regression. Если speculative gain стоил включения chunked prefill, пересмотрите выбор: правильный ответ в 2026 году часто EAGLE-3 без chunked prefill, а не draft model plus chunked prefill, который не compile.

### Числа, которые нужно помнить

- Llama 3.3 70B FP8, H100 SXM5, 128 concurrent, все три включены: 2,200-2,400 tok/s.
- Та же модель, default vLLM (no chunked prefill): ~1,800 tok/s.
- Та же модель, naive PyTorch forward loop: ~600 tok/s.
- KV fragmentation waste при PagedAttention на production load: <4%.
- P99 ITL under mixed load: ~15 ms with chunked prefill, ~50 ms without.

### Как выглядит scheduler

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` — ровно такой loop на stdlib Python с fake token counts и fake forward latency. Запуск показывает, как chunked prefill поддерживает decode sequences во время long prefill.

## Используйте это

`code/main.py` симулирует vLLM-style scheduler с toggleable features. Запустите его, чтобы увидеть:

- `NAIVE` mode: один request за раз, без batching.
- `STATIC` mode: pad and wait, classic batching.
- `CONTINUOUS` mode: iteration-level admission and release.
- `CONTINUOUS + CHUNKED` mode: prefill slices interleaved with decode.

Output показывает total throughput (tokens per virtual second), TTFT mean и P99 ITL. Строка `CONTINUOUS + CHUNKED` должна доминировать на mixed traffic.

## Доведите до результата

Этот урок создает `outputs/skill-vllm-scheduler-reader.md`. По serving config (batch size, KV memory utilization, chunked prefill size, speculative config) он производит scheduler diagnosis, который называет, какой из трех defaults является bottleneck и что tune.

## Упражнения

1. Запустите `code/main.py`. Сравните `STATIC` с `CONTINUOUS` на workload с mixed short and long requests. Откуда возникает throughput gap: prefill efficiency, decode efficiency или tail latency?
2. Измените учебный scheduler, добавив `--max-num-batched-tokens`. Какое правильное значение для H100, running Llama 3.3 70B FP8? (Подсказка: это function of KV block size and number of free blocks, not raw HBM.)
3. Перечитайте vLLM v0.18.0 release notes. Какие combinations of flags mutually exclusive? Перечислите их.
4. Посчитайте KV cache fragmentation waste для trace of 1,000 requests with mean 1,500 output tokens, std 600 tokens, при (a) contiguous per-request allocation at 8192 max, (b) PagedAttention with 16-token blocks.
5. Объясните одним paragraph, почему chunked prefill помогает P99 ITL, но не throughput in isolation. Откуда throughput win появляется на практике?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| PagedAttention | "the KV trick" | Fixed-size block allocator для KV cache; fragmentation <4% |
| Block table | "the page table" | Per-sequence map from logical token position to physical KV block |
| Continuous batching | "dynamic batching, but right" | Admit/release decisions made every decode iteration |
| Chunked prefill | "prefill splitting" | Разбивает long prefill на 512-token slices, interleaved with decode |
| TTFT | "first token time" | Prefill + queue + network; dominated by prefill at long prompts |
| ITL | "inter-token latency" | Time between consecutive decode tokens; dominated by batch size |
| Goodput | "throughput that meets SLO" | Tokens/sec, где every request still hit TTFT and ITL targets |
| V1 scheduler | "the new scheduler" | vLLM's 2026 scheduler; N-gram spec decode is the chunked-prefill-compatible path |
| `--gpu-memory-utilization` | "the memory knob" | Доля HBM, reserved for KV blocks after weights and activations |

## Дополнительное чтение

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — official source on chunked-prefill and speculative-decoding compatibility.
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 release cadence and version-specific behavior.
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — original write-up, который все еще определяет, как думать об allocator.
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — fragmentation analysis and scheduler design.
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — detailed V1 scheduler walkthrough with flame graphs.
