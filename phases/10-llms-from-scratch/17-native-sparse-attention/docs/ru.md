# Native Sparse Attention (DeepSeek NSA)

> На 64k токенов attention съедает 70-80% задержки decode. У каждой open-model lab есть план, как это исправить. NSA от DeepSeek (лучшая статья ACL 2025) — тот вариант, который закрепился: три параллельные attention branches — сжатые coarse-grained tokens, выборочно сохраненные fine-grained tokens и sliding windows для локального контекста — объединяются через обучаемый gate. Он hardware-aligned (удобен для kernels), natively trainable (работает в pre-training, а не прикручивается только на inference) и на 64k decode работает быстрее FlashAttention, сохраняя или превосходя качество full attention. Этот урок строит все три ветки end-to-end и показывает, почему sparsity дифференцируема от начала до конца.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## Цели обучения

- Назвать три NSA attention branches и что захватывает каждая из них.
- Объяснить, почему NSA является "natively trainable", тогда как прежние sparse-attention methods были inference-only.
- Посчитать экономию attention compute у NSA против full attention на 64k context как функцию compression block size и selection top-k.
- Реализовать three-branch combination на stdlib Python для короткой синтетической последовательности и проверить, что gating weights ведут себя разумно.

## Проблема

Full attention при sequence length N стоит `O(N^2)` времени и `O(N)` KV cache на layer. На 64k токенов compute и memory bandwidth становятся катастрофическими. Теоретическая оценка из NSA paper: attention дает 70-80% полной decode latency на 64k. Все downstream — TTFT, tokens/sec, cost per million tokens — доминируется стоимостью attention.

Sparse attention — очевидный ответ. Предыдущие попытки делятся на две группы. Fixed-pattern sparsity (sliding-window, strided, block-local) выбрасывает информацию и проваливается на long-range recall tasks. Inference-time sparsity (KV cache pruning, H2O, StreamingLLM) применяется к модели, обученной на dense attention, и возвращает лишь часть потенциального speedup, потому что модель никогда не училась маршрутизировать информацию через sparse pattern.

Native Sparse Attention (Yuan et al., DeepSeek + PKU + UW, ACL 2025 best paper, arXiv:2502.11089) делает оба: sparsity pattern, который модель учит во время pre-training, реализован как kernel-aligned algorithm, действительно дающий compute savings на inference. Через два года NSA или его прямой потомок будет default attention в каждом frontier long-context model.

## Концепция

### Three parallel branches

Для каждого query NSA запускает attention три раза, по трем разным представлениям KV cache:

1. **Compressed branch.** Токены группируются в blocks размера `l` (обычно 32 или 64). Каждый block сжимается в один summary token через небольшой learned MLP. Query attends over these compressed tokens, получая coarse-grained view всей последовательности.

2. **Selected branch.** По attention scores из compressed branch выбираются top-k blocks, наиболее релевантные текущему query. Fine-grained (несжатые) tokens из этих blocks читаются, и query attends over all of them. Думайте о compressed-branch attention как о routing signal для selection.

3. **Sliding-window branch.** Query attends to наиболее свежим `W` tokens (обычно 512) для локального контекста. Эта branch захватывает structure-heavy short-range patterns (syntax, local coreference), которые две другие могут упустить.

Выходы трех веток объединяются через обучаемый per-position gate:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` — gate weights из небольшого MLP на query. Им не обязательно суммироваться в 1 — они могут независимо взвешивать branches.

### Why this is "natively trainable"

Шаг selection (top-k blocks) дискретен. Дискретные операции ломают gradient flow. Предыдущие работы по sparse attention либо пропускали backprop через selection (ограничивая training), либо использовали continuous relaxations, которые не давали настоящую sparsity на inference.

NSA обходит это: compressed-branch attention — это дифференцируемое coarse-grained attention по всей последовательности. Операция top-k просто повторно использует top attention scores из compressed branch, чтобы выбрать, какие fine-grained blocks загрузить. Gradients проходят через compressed-branch scores (которые влияют и на compressed output, и на selection logic), а вклад selected blocks в final output тоже дифференцируем. Недифференцируемый `top_k` — это no-op для forward computational graph; он только управляет тем, какие blocks загружаются из памяти.

Поэтому NSA можно использовать в pre-training end to end. Модель учится совместно маршрутизировать информацию через три branches, создавая sparse pattern, который на inference действительно дает обещанный speedup.

### Hardware-aligned kernel

NSA kernel спроектирован под современные GPU memory hierarchies. Kernel загружает queries по GQA groups (outer loop), достает соответствующие sparse KV blocks на group (inner loop) и запускает attention в SRAM. Поскольку каждая query group видит те же selected blocks (selection per-query-group, не per-query-head), KV loads амортизируются по group. Arithmetic intensity остается высокой.

В статье сообщают о Triton kernels, работающих в 9x быстрее FlashAttention на 64k decode, причем speedup ratio растет с sequence length. Предоставлены и forward, и backward kernels.

### The compute budget

Пусть `N` — sequence length, `l` — compression block size, `k` — top-k selection count, `w` — sliding window, `b` — selected block size (обычно равен `l`).

- Compressed branch: `O(N/l)` keys на query, значит `O(N * N / l)` всего.
- Selected branch: `O(k * b)` keys на query, значит `O(N * k * b)`.
- Sliding branch: `O(w)` keys на query, значит `O(N * w)`.

Итого: `O(N * (N/l + k*b + w))`.

При `N = 64k, l = 64, k = 16, b = 64, w = 512`: per-query cost равен `1000 + 1024 + 512 = 2536 keys`. Full attention — `64000 keys`. 25x compute reduction.

При `N = 128k, l = 64, k = 16, b = 64, w = 512`: per-query cost равен `2000 + 1024 + 512 = 3536 keys`. Full attention — `128000 keys`. 36x reduction. Выигрыш растет с sequence length — в этом весь смысл.

### How does it compare

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Sliding window only | yes | yes | fails |
| Strided / block-sparse | yes | yes | partial |
| KV pruning (H2O, StreamingLLM) | N/A (inference-time) | yes | partial |
| MoBA (Moonshot) | partial | yes | good |
| NSA | yes (natively) | yes (9x at 64k) | matches full attention |

MoBA (Moonshot, arXiv:2502.13189) была опубликована параллельно и использует похожий подход "три лучше одного", применяя MoE principle к attention blocks. NSA и MoBA — две architecture, которые нужно знать для long-context pre-training 2026 года.

## Практика

`code/main.py` реализует три branches на короткой синтетической последовательности и показывает:

- Compression MLP (для педагогической ясности используется простой mean-pool baseline; настоящий NSA использует learned MLP).
- Top-k block selection, управляемый compressed-branch scores.
- Sliding-window attention на последних `w` tokens.
- Gated combination.
- Compute-count printout с сравнением против full attention.

### Step 1: compress tokens into blocks

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2: compressed-branch attention

Запустите softmax attention query against compressed keys. Compressed-branch scores одновременно являются signal для top-k selection.

### Step 3: top-k block selection

Выберите indices `k` highest-scoring compressed blocks. Загрузите исходные uncompressed tokens из этих blocks и запустите attention на них.

### Step 4: sliding-window attention

Возьмите последние `w` tokens и запустите standard attention against them.

### Step 5: gate + combine

Небольшой MLP на query выдает три gate weights. Final output — weighted sum трех branch outputs.

### Step 6: compute counting

Напечатайте число keys attended per query для каждой branch и total. Сравните с `N` (full attention). На синтетике 1024 tokens с `l = 32, k = 4, w = 128` NSA видит `32 + 128 + 128 = 288` keys на query против 1024 для full attention — в 3.5x меньше.

## Использование

NSA поставляется в собственном long-context pre-training pipeline DeepSeek. Статус интеграции в публичных inference stacks по состоянию на апрель 2026:

- **DeepSeek internal**: native, опубликованные weights используют NSA или его successor DSA (Deepseek Sparse Attention).
- **vLLM**: экспериментальная поддержка NSA в разработке для DeepSeek-V3.x weights.
- **SGLang**: NSA benchmarks опубликованы; production path следует за vLLM.
- **llama.cpp / CPU**: не поддерживается; overhead от kernel decomposition не окупается на CPU throughput.

Когда использовать NSA:

- Pre-training или continued-training run, нацеленный на 64k+ context с серьезным compute budget.
- Inference собственных long-context checkpoints DeepSeek. Weights NSA-native.

Когда не использовать:

- Serving существующей pre-trained model с dense attention. NSA нельзя retrofit-ить без continued training.
- Контекст меньше 16k. Overhead трех branches доминирует над savings.
- Batch-1 interactive chat. Latency-sensitive decode выигрывает, но только на длинных контекстах.

## Результат

Этот урок создает `outputs/skill-nsa-integrator.md`. По спецификации long-context pre-training run он строит NSA integration plan: compression block size, top-k, sliding window, gate MLP width, kernel choice и конкретные long-context evals, которые оправдали бы смену architecture.

## Упражнения

1. Запустите `code/main.py` на синтетике 1024 tokens. Sweep `(l, k, w)` по трем presets и напечатайте compute counts. Найдите preset с минимальным key-count per query при сохранении 95% recall против full attention на needle-in-haystack test.

2. Замените mean-pool compressor на tiny learned MLP (2-layer, hidden 32). Обучите его на synthetic task, где signal — это average of a block. Измерьте perplexity gap против mean-pool baseline на held-out data.

3. Реализуйте gate MLP. Он принимает query и выводит три scalars. Покажите, что gate ведет себя разумно: почти uniform weighting на random queries, большой вес selected branch, когда query попадает в far-back block.

4. Посчитайте KV cache memory budget для NSA-enabled 70B model на 128k context. KV heads равны 8, head dim 128, BF16. Сравните с full attention и MLA (Phase 10 · 14 показывал числа MLA). Найдите sequence length, где fine-grained branch KV cache у NSA равен full attention.

5. Прочитайте Section 4 статьи NSA (arXiv:2502.11089) и объясните в трех предложениях, почему attention scores compressed branch повторно используются для top-k selection вместо вычисления отдельного routing score. Свяжите ответ с gradient flow.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Compressed branch | "Coarse view" | Attention по block-averaged keys, дающее global context за O(N/l) keys на query |
| Selected branch | "Top-k blocks" | Fine-grained attention по `k` blocks с highest compressed-branch scores |
| Sliding window | "Local context" | Attention по последним `W` tokens для short-range patterns |
| Native trainability | "Pre-train with the sparsity on" | Sparsity pattern учится во время pre-training, а не прикручивается на inference |
| Compression block size l | "Group size for coarse view" | Сколько tokens объединяются в один summary; типично 32-64 |
| Top-k | "Blocks to keep" | Число compressed blocks, чьи uncompressed tokens читаются; типично 16 |
| Sliding window W | "Local attention radius" | Обычно 512; короче ухудшает local coherence, длиннее тратит compute |
| Branch gate | "How to mix the three" | Per-position MLP output, который взвешивает вклады трех branches |
| Hardware alignment | "Kernel-friendly sparsity" | Sparse pattern выбран так, чтобы реальный GPU kernel достигал теоретического speedup |
| DSA | "NSA's successor" | Deepseek Sparse Attention, architecture, последовавшая за NSA в lineage DeepSeek |

## Дополнительное чтение

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — статья
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — architecture family, на которую нацелен NSA
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — параллельная работа, MoE-style attention over blocks
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — истоки sliding-window
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — inference-time sparsity baseline, который улучшает NSA
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — full-attention baseline, который NSA kernels обгоняют на 64k
