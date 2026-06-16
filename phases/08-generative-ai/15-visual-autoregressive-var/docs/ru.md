# Visual Autoregressive Modeling (VAR): Next-Scale Prediction

> Diffusion models sample iteratively in time (denoising steps). VAR sample-ит итеративно по масштабу: предсказывает 1x1 token, затем 2x2, затем 4x4, до final resolution, и каждый scale conditioning on previous. Статья 2024 года показала, что VAR следует GPT-style scaling laws для image generation и обгоняет DiT при том же compute budget. Этот урок строит core mechanism.

**Тип:** Сборка
**Языки:** Python (with PyTorch)
**Предварительные требования:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Время:** ~90 минут

## Цели обучения

- Объяснять next-scale prediction: генерацию изображения как пирамиды токен-сеток от 1×1 до полного разрешения.
- Строить многомасштабный residual-VQ токенизатор и scale-ordered маску внимания — каузальную между масштабами и параллельную внутри масштаба.
- Объяснять, почему next-scale побеждает next-token (от грубого к точному, параллельность внутри масштаба, отсутствие смещения порядка) и как VAR воспроизводит законы масштабирования в стиле GPT.

## Проблема

Autoregressive generation доминировала в language modeling, потому что масштабируется предсказуемо: больше compute, больше parameters, ниже perplexity, лучше outputs. До 2024 года у image generation было две основные AR-попытки: PixelRNN/PixelCNN (pixel-by-pixel) и DALL-E 1 / Parti / MuseGAN (token-by-token on VQ-VAE codes).

Обе страдали от generation-order problem. Pixels и tokens расположены в 2D grid, но AR model должна обходить их в 1D raster order. Ранний corner pixel не знает, чем eventually станет image. Generation quality масштабировалось хуже, чем GPT-on-text, и не достигало diffusion-model quality при matched compute.

VAR исправляет generation-order problem, меняя то, что генерируется. Вместо prediction image tokens one by one in space, VAR predicts whole image at increasing resolutions. Step 1: predict 1x1 token (overall image "summary"). Step 2: predict 2x2 grid of tokens (coarser features). Step 3: predict 4x4 grid. Step K: predict final (H/8)x(W/8) grid.

Каждый scale attends to all previous scales (causally in "scale order") и parallel внутри своего scale. Order problem исчезает: whole image at scale k производится за один transformer pass.

## Концепция

### VQ-VAE Multi-Scale Tokenizer

VAR нужен **multi-scale discrete tokenizer**. Для image x он производит sequence progressively higher-resolution token grids:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Каждый z_k использует same codebook (typical size 4096-16384). Tokenization на каждом scale не independent — она обучается так, чтобы сумма residuals на каждом scale reconstruct f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Это variant **residual VQ**. Scale k захватывает то, что missed scales 1..k-1. Decoder берет sum of all scale embeddings и produces image.

Multi-scale VQ tokenizer обучается один раз (как VQGAN), затем frozen. Вся generative work делается autoregressive model сверху.

### Next-Scale Prediction

Generative model — transformer, который видит tokens всех previous scales и predicts tokens at next scale.

Input sequence structure:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embeddings encode both scale index and spatial position within scale. Attention causal in scale order: token at scale k, position (i, j) can attend to all tokens at scales 1..k and to tokens at scale k itself that come earlier in intra-scale order (VAR uses fixed positional attention with no intra-scale causality — all positions within a scale are predicted in parallel).

Training loss: на каждом scale k predict tokens z_k given all prior-scale tokens. Cross-entropy loss on discrete VQ codes. Та же структура, что GPT, но "sequence" теперь scale-structured.

### Generation

At inference:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Для K = 10 scales generation — 10 transformer forward passes. Каждый pass производит весь scale parallel — без per-token autoregression внутри scale. Для 256x256 image это примерно 10 passes vs 28-50 у DiT.

### Почему Next-Scale выигрывает у Next-Token

Три structural wins:
1. **Coarse-to-fine aligns with natural image statistics.** Human visual perception и image datasets показывают scale-dependent regularities: low-frequency structure stable and predictable; high-frequency detail conditional on low-frequency content. Next-scale prediction exploits this.
2. **Parallel generation within scale.** В отличие от GPT-style token AR, VAR производит все tokens scale за один step. Effective generation length log-scale instead of linear.
3. **No generation order bias.** Tokens at scale k видят весь scale k-1; нет "left-of" или "above" bias, заставляющего early tokens commit before late context available.

### Scaling Law

Tian et al. показали, что VAR следует power-law scaling curve для FID on ImageNet — как GPT для perplexity. Doubling parameters или compute reliably halves error. Это была первая image-generative model с таким чистым scaling behavior, как у language models. В результате VAR-scale predictions становятся predictable from compute, а не empirical guesses per architecture.

### Relationship to Diffusion

VAR и diffusion делят одну data-compression story: оба разбивают generation problem на sequence easier subproblems.

- Diffusion: gradually add noise, learn to undo one step.
- VAR: gradually add resolution, learn to predict the next scale.

Это разные axes через problem. Оба дают tractable conditional distributions. Empirically VAR faster at inference (fewer passes, all parallel within scale) and matches or beats DiT on class-conditional ImageNet. Text-conditional VAR (VARclip, HART) — active research direction.

## Практика

В `code/main.py` вы:
1. Построите tiny **multi-scale VQ tokenizer** на synthetic "image" data (2D Gaussian rings).
2. Обучите **VAR-style transformer** на next-scale-predict tokens.
3. Sample через 4 calls transformer (4 scales) and decoding.
4. Проверите, что scale-ordered training делает generation parallel within scale.

Это toy implementation. Смысл — увидеть scale-structured attention mask и parallel-within-scale generation в работе.

## Запуск в продукт

Этот урок производит `outputs/skill-var-tokenizer-designer.md` — skill for designing multi-scale tokenizer: number of scales, scale ratios, codebook size, residual sharing, decoder architecture.

## Упражнения

1. **Scale count ablation.** Train VAR with 4, 6, 8, 10 scales. Measure reconstruction quality vs number of autoregressive passes. More scales = finer residuals = better quality but more passes.

2. **Codebook size.** Train tokenizers with codebook sizes 512, 4096, 16384. Larger codebooks give better reconstruction but harder prediction. Найдите knee.

3. **Parallel-within-scale check.** Для trained VAR явно измерьте attention pattern. Within scale k, does the model attend to cross-scale positions but not intra-scale? Verify mask implementation.

4. **VAR vs DiT scaling.** Для same ImageNet class-conditional task train VAR and DiT at matched param budgets (e.g., 33M, 130M, 458M). Plot FID vs compute. VAR should pull ahead of DiT at each size — reproduce paper result at small scale.

5. **Text conditioning.** Extend VAR to take text embedding (CLIP pooled) как extra conditioning input via adaLN. This is the HART recipe. How much does FID improve on text-aligned sampling?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | Model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale k can attend to all of scales 1..k but not scales k+1..K |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both scale index and spatial coordinates within scale |

## Дополнительное чтение

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — статья VAR, canonical reference
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, diffusion comparison baseline
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, tokenizer family, которую extends multi-scale tokenizer VAR
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, foundation of discrete image tokenization
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — text-conditional VAR
