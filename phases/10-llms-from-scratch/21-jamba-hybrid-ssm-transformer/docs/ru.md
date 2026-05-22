# Jamba — Hybrid SSM-Transformer

> State space models (SSMs) и transformers хотят разного. Transformers покупают качество через attention с quadratic cost. SSMs покупают linear-time inference и constant memory через recurrence, но отстают по quality. Jamba от AI21 (март 2024) и Jamba 1.5 (август 2024) помещают их в одну модель: 1 Transformer layer на каждые 7 Mamba layers, MoE на каждом втором block и context window 256k, помещающийся на один 80GB GPU. Mamba-3 (ICLR 2026) усиливает SSM-сторону complex-valued state spaces и MIMO projections. Этот урок читает обе architectures end to end и объясняет, почему hybrid recipe пережил три года scaling, тогда как pure-SSM и pure-Transformer long-context attempts — нет.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## Learning Objectives

- Объяснить три primitives в Jamba block — Transformer layers, Mamba layers, MoE — и рецепт interleaving 1:7:even.
- Сформулировать на высоком уровне, как выглядит recurrence SSM и почему она дает constant-memory inference.
- Посчитать KV cache footprint Jamba model на 256k context и сравнить с pure-Transformer model.
- Назвать три innovations Mamba-3 (exponential-trapezoidal discretization, complex-valued state update, MIMO) и проблему, на которую нацелена каждая.

## The Problem

Attention квадратично по sequence length. State space models линейны. Эта разница накапливается: на 256k tokens attention map Transformer содержит 65B entries на head; recurrent state SSM имеет фиксированный размер независимо от sequence length.

Pure-SSM models (Mamba, Mamba-2) совпадают с Transformer perplexity на малых scales, но отстают на state-tracking tasks и проваливаются на некоторых категориях in-context retrieval. Интуиция: SSMs сжимают history в fixed state, и когда history длинная, information leaks. Attention помнит все точно, но платит quadratic cost.

Очевидное исправление: использовать оба. Поставить Transformer layers там, где важен exact recall. Использовать SSM layers в остальных местах. Настроить ratio. Jamba — первая production-grade model, которая shipped this hybrid recipe at scale (52B total, 12B active, 256k context, single 80GB GPU). Jamba 1.5 расширяет family до 398B total / 94B active. Mamba-3 (ICLR 2026) — current-best pure-SSM baseline, вокруг которого можно перестраивать hybrids.

Этот урок читает все три papers и дает mental model для "pick the right ratio."

## The Concept

### An SSM in one page

State space model обрабатывает sequence `x_1, ..., x_N` через fixed-size state `h`:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

На каждом шаге state evolves через linear dynamics `A`, принимает input `B x_t` и emits output `C h_t`. `A, B, C` могут обучаться. Критическое свойство: чтобы посчитать `y_t`, нужны только `h_{t-1}` и `x_t`, а не более ранние `x`. Memory constant. Inference O(1) на token.

Трюк для modeling quality — структура `A`. S4 (Gu 2021) использовал highly structured matrix, которую можно было эффективно вычислять как long convolution во время training. Mamba (Gu, Dao 2023) заменил fixed `A, B, C` на data-dependent ones (the "selective" part). Mamba-2 (2024) еще больше упростил structure. Mamba-3 (2026) заново добавляет complexity в конкретных местах.

Ключевое свойство: для decoder LLM SSM layer — drop-in replacement для attention layer, с fixed-size per-layer state вместо растущего KV cache.

### The Jamba block

Jamba block чередует layers по двум числам:

- `l`: attention-to-Mamba ratio. Jamba использует `l = 8`, то есть 1 Transformer layer на 7 Mamba layers (7 Mamba + 1 Attention = 8 layers per group).
- `e`: MoE frequency. Jamba использует `e = 2`, то есть каждый второй layer применяет MoE.

Layer sequence внутри block:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

Каждый Jamba block — 8 layers. На глубине 4 blocks (32 layers total) получается 28 Mamba и 4 Attention layers. 16 из них используют MoE.

### Why the 1:7 ratio

AI21 провела ablations: какой ratio attention-to-Mamba дает лучшую perplexity-per-parameter И in-context recall на long-context evals?

- Слишком много attention (1:1): качество растет, но memory и speed деградируют.
- Слишком мало attention (1:15): memory отличная, но in-context retrieval проваливается.
- Sweet spot: 1:7 или 1:8.

Интуиция: Transformer layers отвечают за exact recall и state tracking. Mamba layers делают дешевую основную массу processing.

### Positional encoding

Mamba layers сами position-aware (через recurrence). Attention layers в исходных Mamba-based hybrids не использовали RoPE — SSM layers давали position info. Jamba 1.5 добавляет RoPE в attention layers для longer-context generalization, post-hoc refinement на основе empirical long-context evaluation.

### The memory budget

Для формы Jamba-1 (32 layers: 28 Mamba + 4 Attention, hidden 4096, 32 attention heads):

- KV cache (только attention layers): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` при 256k BF16. Вклад дают только 4 attention layers.
- SSM state: `28 * hidden * state_size` на token prefix, но это fixed-size per layer, не scaling with sequence length. Типичный Mamba state — 16 на feature, hidden 4096: `28 * 4096 * 16 * 2 = 3.7 MB` total.

Сравните с pure Transformer в 32 layers, same hidden, full MHA at 32 heads: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` при 256k BF16. 8x reduction в KV cache. Даже против GQA(8) baseline, который использует большинство 2024 models (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), Jamba hybrid 1:7 при 16 GB все равно в 2x меньше.

Вот что AI21 имеет в виду под "256k context on a single 80GB GPU." KV cache pure Transformer с full-MHA не поместился бы; даже GQA baseline почти не оставляет места для weights and activations; Jamba помещает.

### Mamba-3: the pure-SSM baseline in 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) вводит три innovations на pure-SSM side:

1. **Exponential-trapezoidal discretization.** Заменяет Euler-method discretization в Mamba-2 на более expressive recurrence. Convolution-like operation применяется к state-input внутри core recurrence, а не как outer convolution на `x_t`.

2. **Complex-valued state update.** Previous Mambas упростили state matrix от complex (S4) к real diagonal (Mamba) и scaled identity (Mamba-2). Mamba-3 возвращает complex values — эквивалент data-dependent rotary embedding на state. Это восстанавливает state-tracking capabilities, которые стоили previous real-valued simplifications.

3. **Multi-input multi-output (MIMO) projections.** Вместо per-feature scalar projections используются matrix-valued projections. Улучшает modeling power и inference-time hardware utilization без увеличения decode latency.

На 1.5B parameters Mamba-3 улучшает average downstream accuracy на 0.6 points относительно Gated DeltaNet; MIMO variant добавляет еще 1.2, всего 1.8-point gain. При том же state size Mamba-3 совпадает с Mamba-2 с половиной state.

Mamba-3 пока не shipped в production hybrid at scale — но она очевидный кандидат для SSM side следующей Jamba-class model.

### When to reach for a hybrid

Hybrids выигрывают, когда:

- Context достаточно длинный, чтобы pure Transformer KV cache стал болезненным (64k+).
- Tasks смешивают short-range structure (хорошо для SSM) с long-range recall (нужен Transformer).
- Нужно deploy на single-GPU memory budgets, где один Transformer KV cache не поместился бы.

Hybrids проигрывают, когда:

- Context короткий (меньше 16k). SSM overhead тратится зря; pure Transformer нормален.
- Tasks требуют everywhere-to-everywhere attention (deep reasoning, multi-document cross-reference). Sparse attention layers в hybrid вредят.
- Вы масштабируетесь к trillion-parameter frontier models. Pure-Transformer + MLA + MoE (стиль DeepSeek-V3) сейчас выигрывает capability race.

### The competitive landscape

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

Ландшафт 2026: pure-Transformer MoE доминирует на frontier, но hybrids владеют нишей 256k+ context. State-tracking wins Mamba-3 могут сдвинуть hybrid ratios ниже (больше SSM, меньше attention) в следующем поколении.

## Use It

`code/main.py` — memory calculator для hybrid architectures. По SSM-Transformer ratio и hidden-size / layer-count config он считает:

- KV cache на target context.
- SSM state memory.
- Total memory at context N для набора model shapes.

Calculator поддерживает:

- Pure-Transformer baseline (KV cache grows with N).
- Jamba-style 1:7 hybrid.
- Pure-SSM (no KV cache at all).

Числа для published shapes взяты напрямую из papers Jamba-1 и Jamba-1.5, а для hypothetical variants экстраполированы.

Integration considerations для real deployment:

- Большинство production inference servers (vLLM, SGLang) поддерживают Jamba и Mamba. Проверяйте specific version.
- На 256k context memory advantage Jamba проявляется в concurrent-request throughput. На той же VRAM помещается больше Jamba sequences, чем Transformer sequences.
- Mamba-3 как standalone model пока не shipped in production — research preview at 1.5B.

## Ship It

Этот урок создает `outputs/skill-hybrid-picker.md`. По workload specification (context length profile, task mix, memory budget) он рекомендует pure Transformer, Jamba-style hybrid или pure SSM, с явным reasoning о memory and quality tradeoffs.

## Exercises

1. Запустите `code/main.py`, чтобы посчитать KV cache на 256k context для 32-layer pure Transformer (hidden 4096, 32 heads) и для Jamba-1 hybrid той же формы. Проверьте ~8x memory reduction, заявленный paper AI21.

2. Измените calculator, чтобы смоделировать 1:3 hybrid (4 Mamba : 1 Attention) и 1:15 hybrid (14 Mamba : 1 Attention). Постройте KV cache vs ratio. При каком ratio KV cache равен SSM state memory?

3. Прочитайте Section 3 paper Jamba (arXiv:2403.19887). Объясните, почему AI21 использует Mamba-1, а не Mamba-2, несмотря на то что Mamba-2 быстрее. Hint: hybrid ablation section это документирует.

4. Посчитайте parameter overhead MoE-every-other-layer в Jamba 1.5 Large (398B total, 94B active). Сравните active ratio с DeepSeek-V3 (37B/671B) и объясните, почему architecture Jamba поднимает active ratio выше.

5. Прочитайте Section 3 paper Mamba-3 (arXiv:2603.15569). В трех предложениях объясните, почему complex-valued state update эквивалентен data-dependent rotary embedding. Свяжите ответ с выводом RoPE в Phase 7 · Lesson 04.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State space model (SSM) | "Recurrence with a fixed state" | Layer с learned recurrence `h_t = A h_{t-1} + B x_t`; constant memory per token |
| Selective SSM | "Mamba's trick" | Data-dependent A, B, C parameters, дающие model gating-like selectivity at linear time |
| Attention-to-Mamba ratio | "How many attention layers" | В Jamba `l = 8` означает 1 attention layer на 7 Mamba layers |
| Jamba block | "The 8-layer group" | Один attention + seven Mamba + MoE на alternate positions |
| SSM state | "The hidden buffer" | Fixed-size per-layer state, заменяющий KV cache для Mamba layers |
| 256k context | "Jamba's flagship number" | Sequence length, который Jamba-1 помещает на single 80GB GPU; pure Transformer не может при этом размере |
| Mamba-3 | "2026 pure SSM" | Current-best pure-SSM architecture с complex state + MIMO; baseline, вокруг которого hybrids rebuild |
| MIMO | "Multi-input multi-output" | Innovation Mamba-3 с matrix-valued projections вместо scalar per-feature |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | Более expressive recurrence, subsuming Euler-method discretization Mamba-2 |
| Hybrid architecture | "Mix attention and SSM" | Любая model, interleaving Transformer and SSM layers; Jamba — production archetype |

## Further Reading

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — original Jamba paper, ratio ablations, claim 256k context
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — scaled-up family, 398B/94B и 12B/52B public releases
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — selective SSM paper, на которой строится Jamba
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — simplified structured-state-space successor
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — complex-valued state, MIMO, frontier pure-SSM 2026
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — paper S4, starting point genealogy SSM для LLMs
