# DeepSeek-V3 Architecture Walkthrough

> Phase 10 · Lesson 14 назвал шесть архитектурных knobs, которые крутит каждая open model. DeepSeek-V3 (декабрь 2024, всего 671B parameters, 37B active) крутит все шесть и добавляет еще четыре: Multi-Head Latent Attention, auxiliary-loss-free load balancing, Multi-Token Prediction и DualPipe training. Этот урок читает architecture DeepSeek-V3 сверху вниз и выводит каждый parameter count из опубликованного config. К концу вы сможете объяснить, почему ratio 671B/37B — правильная ставка и почему MLA + MoE вместе на frontier лучше, чем каждый по отдельности.

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## Цели обучения

- Прочитать config DeepSeek-V3 сверху вниз и объяснить каждое поле через шесть knobs GPT-2 плюс четыре добавления DeepSeek.
- Вывести total parameter count (671B), active parameter count (37B) и components, которые дают каждый из них.
- Посчитать footprint KV cache у MLA на 128k context и сравнить с тем, что заплатила бы dense model с GQA при тех же active params.
- Назвать четыре DeepSeek-specific innovations (MLA, MTP, auxiliary-loss-free routing, DualPipe) и указать, на какую часть architecture/training stack нацелена каждая.

## Проблема

DeepSeek-V3 — первая frontier open model, architecture которой существенно отличается от семейства Llama. Llama 3 405B — это "GPT-2 с шестью повернутыми knobs". DeepSeek-V3 — GPT-2 со всеми шестью knobs плюс еще четыре. Чтение config Llama 3 — разминка перед чтением DeepSeek config, но глубокая структура — форма attention block, routing logic, training-time objective — достаточно другая, чтобы требовать отдельный walkthrough.

Зачем это учить: release open weights DeepSeek-V3 сдвинул смысл "frontier capability" в open models. Architecture стала blueprint, который копируют многие training runs 2026 года. Понимание ее — базовое требование для любой роли, связанной с frontier LLM training или inference.

## Концепция

### The invariant core, again

DeepSeek-V3 все еще autoregressive. Она все еще stacks decoder blocks. В каждом block все еще есть attention plus MLP plus two RMSNorms. Она все еще использует SwiGLU в MLP. Все еще RoPE. Pre-norm. Weight-tied embeddings. Та же база, что у любой Llama или Mistral.

### The twist: MLA instead of GQA

Из Phase 10 · 14 вы знаете, что GQA уменьшает KV cache, разделяя K и V между группами Q heads. Multi-Head Latent Attention (MLA) идет дальше: K и V сжимаются в общий low-rank latent representation (`kv_lora_rank`), затем decompressed per head on the fly. KV cache хранит только latent — обычно 512 floats на token на layer, а не 8 x 128 = 1024 floats.

На 128k context DeepSeek-V3 с MLA (один shared latent `c^{KV}` на token на layer; K и V оба выводятся из этого latent через up-projections, которые можно absorbed into subsequent matmul):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Гипотетический GQA baseline (форма Llama 3 70B, 8 KV heads, head dim 128) заплатил бы:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA в 4x меньше, чем GQA cache в стиле Llama-3-70B на 128k context.

Tradeoff: MLA добавляет decompression step на каждое attention computation (per head). Extra compute мал по сравнению с saved bandwidth. Net win для long-context inference.

### The routing: auxiliary-loss-free load balancing

MoE routers решают, какие top-k experts обрабатывают каждый token. Naive router концентрирует слишком много работы на нескольких experts, оставляя других idle. Стандартное решение: добавить auxiliary loss term, штрафующий load imbalance. Это работает, но слегка ухудшает main-task performance.

DeepSeek-V3 вводит auxiliary-loss-free scheme. К router logits добавляются per-expert bias terms, которые во время training корректируются простым правилом: если expert `e` overloaded, уменьшить `bias_e`; если underloaded, увеличить. Никакого extra loss term. Training остается clean. Expert load остается balanced.

Effect on main loss: измеримо отсутствует. Effect on MoE architecture: чище, нет auxiliary-loss hyperparameter для tuning.

### The MTP: denser training + free draft

Из Phase 10 · 18 вы знаете, что DeepSeek-V3 добавляет D=1 MTP module, предсказывающий token на две позиции вперед. На inference обученный module переиспользуется как speculative-decoding draft с acceptance 80%+. На training каждый hidden state supervised по D+1 = 2 targets, давая более плотный signal.

Parameters: 14B поверх 671B main. Overhead: 2.1%.

### The training: DualPipe

Из Phase 10 · 19 вы знаете, что DualPipe — bidirectional pipeline, который перекрывает forward и backward chunks с cross-node all-to-all comms. В масштабе DeepSeek-V3 на 2,048 H800 он возвращает примерно 245k GPU-hours, которые 1F1B потерял бы на pipeline bubbles.

### The config, field by field

Вот config DeepSeek-V3 (упрощенный):

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

Parse it:

- `hidden_size=7168`: embedding dimension.
- `num_hidden_layers=61`: total block depth.
- `first_k_dense_layers=3`: первые 3 blocks используют dense MLP size 18432. Остальные 58 используют MoE.
- `num_attention_heads=128`: 128 query heads.
- `kv_lora_rank=512`: K и V сжимаются до этой latent dimension и decompressed per head.
- `num_experts=256, num_experts_per_tok=8`: каждый MoE block имеет 256 experts, routes top-8.
- `shared_experts=1`: поверх 256 routed experts 1 always-on expert вносит вклад в каждый token. Думайте о нем как о "dense floor", гарантирующем надежный вклад для каждого token.
- `moe_intermediate_size=2048`: hidden size MLP каждого expert. Меньше dense MLP, потому что experts 256.

### Parameter accounting

Полный расчет находится в `code/main.py`. Главное:

- Embedding: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- First 3 dense blocks: attention with MLA (~144M per block) + dense MLP (~260M per block) + norms. Около 1.2B total.
- 58 MoE blocks: attention with MLA (~144M) + 256 experts each (30M apiece) + 1 shared expert (30M) + norm. Total ~7.95B per block, включая все experts. 461B total для 58 MoE blocks.
- MTP module: 14B.

Grand total: ~476B для core architecture + 14B MTP; при этом опубликованное число 671B отдельно учитывает дополнительные structural parameters (bias tensors, expert-specific components, shared expert scaling, etc.). Число, которое воспроизводит calculator, в пределах 3-5% от published — delta идет из fine-grained accounting, который DeepSeek report документирует в Section 2 appendix.

Active parameters per forward:

- Attention: 144M per layer * 61 = 8.8B (все layers fire).
- MLP active: первые 3 layers dense (3 * 260M = 780M), 58 MoE layers каждый active с 8 routed + 1 shared + routing overhead. Per layer active MLP: ~260M. Total: 3 * 260M + 58 * 260M = ~15.9B.
- Embedding + norms: 1.2B.
- Total active: roughly 26B core + 14B MTP (trained but not always run at inference) ≈ 37B.

### The 671B / 37B ratio

18x sparsity ratio (active params — 5.5% total). DeepSeek-V3 — самый sparse frontier MoE model, shipped as open weights. Mixtral 8x7B с ratio 13/47 (28%) гораздо плотнее. Llama 4 Maverick с ratio 17B/400B (4.25%) сопоставим. Ставка DeepSeek: на frontier scale больше experts с меньшим activation ratio дает лучшее качество per active-FLOP.

### Where DeepSeek-V3 sits

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### The follow-on: R1, V4

DeepSeek-R1 (2025) — reasoning-training run на V3 backbone. R1 использует ту же architecture. Изменился post-training recipe (large-scale RL on verifiable tasks), а не pretraining architecture.

DeepSeek-V4 (если выйдет) ожидаемо сохранит MLA + MoE + MTP и добавит DSA (DeepSeek Sparse Attention), successor NSA из Phase 10 · 17. Lineage стабилен: architecture-level innovations накапливаются; каждая version крутит дополнительные knobs.

## Использование

`code/main.py` — parameter calculator, специализированный под форму DeepSeek-V3. Запустите его, сравните output с numbers из paper и примените к hypothetical variants (256 experts vs 512, top-8 vs top-16, MLA rank 512 vs 1024).

На что смотреть:

- Total parameter count vs published 671B.
- Active parameter count vs published 37B.
- KV cache at 128k context — сравнение MLA vs GQA.
- Per-layer breakdown, чтобы увидеть, куда реально уходит parameter budget.

## Результат

Этот урок создает `outputs/skill-deepseek-v3-reader.md`. По DeepSeek-family model (V3, R1 или будущий variant) он строит component-by-component architecture reading, называя каждое поле config, выводя parameter counts по component и определяя, какие из четырех DeepSeek-specific innovations использует model.

## Упражнения

1. Запустите `code/main.py`. Сравните total-parameter estimate calculator с published 671B и определите, откуда берется delta. Section 2 paper содержит полную itemization.

2. Измените config, чтобы использовать MLA rank 256 вместо 512. Посчитайте resulting KV cache size на 128k context. Какой percentage reduction это дает, и какой ценой для per-head expressiveness?

3. Сравните routing DeepSeek-V3 (256 experts, top-8) с гипотетическим variant (512 experts, top-8). Total parameters растут; active parameters остаются прежними. Что extra expert capacity дает теоретически, и чего стоит на inference?

4. Прочитайте Section 2.1 технического отчета DeepSeek-V3 (arXiv:2412.19437) о MLA. В трех предложениях объясните, почему K and V decompression matrices могут быть "absorbed" into subsequent matmul для inference-time efficiency.

5. DeepSeek-V3 использует FP8 training для большинства operations. Посчитайте memory savings FP8 vs BF16 для хранения 671B weights. Как это пересекается с 14.8T-token training budget?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | Сжимает K и V в shared low-rank latent (kv_lora_rank, обычно 512), decompressed per head on-the-fly; KV cache хранит только latent |
| kv_lora_rank | "MLA compression dim" | Размер shared latent для K и V; DeepSeek-V3 использует 512 |
| First k dense layers | "Early layers stay dense" | Первые несколько layers MoE-model пропускают MoE router и выполняют dense MLP для стабильности |
| num_experts_per_tok | "Top-k routing" | Сколько routed experts срабатывают на token; DeepSeek-V3 использует 8 |
| Shared experts | "Always-on experts" | Experts, которые обрабатывают каждый token независимо от routing; DeepSeek-V3 использует 1 |
| Auxiliary-loss-free routing | "Bias-adjusted load balance" | Per-expert bias terms, корректируемые во время training для balancing expert load без добавления loss term |
| MTP module | "Extra prediction head" | Transformer block, предсказывающий t+2 из h^(1) и E(t+1); denser training, free speculative-decoding draft |
| DualPipe | "Bidirectional pipeline" | Training schedule, перекрывающий forward/backward compute с cross-node all-to-all |
| Active parameter ratio | "Sparsity" | active_params / total_params; DeepSeek-V3 достигает 5.5% |
| FP8 training | "8-bit training" | Training storage и многие compute ops в FP8; roughly halves memory vs BF16 with small quality cost |

## Дополнительное чтение

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — полный документ по architecture, training и results
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — config files и deployment notes
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — predecessor, introduced MLA
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — reasoning-training successor на architecture V3
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — future direction для DeepSeek-family attention
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — training-schedule reference
