# Оценка — FID, CLIP Score, Human Preference

> Каждый leaderboard генеративных моделей цитирует FID, CLIP score и win rate из human-preference arena. У каждого числа есть failure mode, который настойчивый researcher может game. Если вы не знаете failure modes, вы не отличите реальное улучшение от gaming run.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 01 (Taxonomy), Фаза 2 · 04 (Evaluation Metrics)
**Время:** ~45 минут

## Проблема

Generative model оценивают по *sample quality* и *conditioning adherence*. Ни у того, ни у другого нет closed-form measure. Ваша model должна render 10,000 images; что-то должно назначить им numbers; вы должны доверять numbers across model families, resolutions и architectures. Три metrics прошли gauntlet 2014-2026:

- **FID (Fréchet Inception Distance).** Distance between two distributions — real и generated — в feature space Inception network. Lower is better.
- **CLIP score.** Cosine similarity между CLIP-image embedding generated image и CLIP-text embedding prompt. Higher is better. Измеряет prompt adherence.
- **Human preference.** Столкнуть две models head-to-head on same prompt, дать humans (или GPT-4-class model) выбрать better one, aggregate to Elo score.

Также встретите: IS (inception score, largely retired), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Каждый исправляет один failure предыдущего.

## Концепция

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — sample quality

Heusel et al. (2017). Steps:

1. Extract Inception-v3 features (2048-D) for N real images and N generated.
2. Fit Gaussian to each pool: compute mean `μ_r, μ_g` and covariance `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Interpretation: Fréchet distance between two multivariate Gaussians in feature space. Lower = more similar distributions.

Failure modes:
- **Biased on small N.** FID — mean-squared по feature distribution; small N under-estimates covariance, gives falsely low FID. Always use N ≥ 10,000.
- **Inception-dependent.** Inception-v3 trained on ImageNet. Domains far from ImageNet (faces, art, text images) produce meaningless FID. Use domain-specific feature extractor.
- **Gaming.** Overfitting to Inception prior дает low FID without visual quality improvement. Beat it with CMMD (below).

### CLIP score — prompt adherence

Radford et al. (2021). For generated image + prompt:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Average across 30k generated images → scalar comparable between models.

Failure modes:
- **CLIP's own blind spots.** У CLIP weak compositional reasoning ("a red cube on a blue sphere" часто fails). Models can rank well on CLIP score without really following complex prompts.
- **Short prompt bias.** Short prompts имеют больше CLIP-image matches in the wild. Longer prompts have lower CLIP scores mechanically.
- **Prompt gaming.** Добавление "high quality, 4k, masterpiece" в prompt inflates CLIP score without improving image-text binding.

CMMD (Jayasumana et al., 2024) исправляет часть этого: uses CLIP features instead of Inception, maximum-mean discrepancy instead of Fréchet. Better at detecting subtle quality differences.

### Human preference — ground truth

Выберите pool of prompts. Generate with model A and model B. Покажите pairs humans (или strong LLM judge). Aggregate wins into Elo or Bradley-Terry score. Benchmarks:

- **PartiPrompts (Google)**: 1,600 diverse prompts, 12 categories.
- **HPSv2**: 107k human annotations, widely used as automated proxy.
- **ImageReward**: 137k prompt-image preference pairs, MIT-licensed.
- **PickScore**: trained on Pick-a-Pic 2.6M preferences.
- **Chatbot-Arena-style image arenas**: https://imagearena.ai/ and others.

Failure modes:
- **Judge variance.** Non-experts have different preferences than experts. Use both.
- **Prompt distribution.** Cherry-picked prompts favor one family. Always document.
- **LLM-judge reward hacking.** GPT-4-judge gets fooled by pretty-but-wrong outputs. Triangulate with human.

## Совместное использование

Production eval report должен включать:

1. FID on 10-30k samples against held-out real distribution (sample quality).
2. CLIP score / CMMD on same samples vs prompts (adherence).
3. Win rate in blinded arena vs previous model (overall preference).
4. Failure mode analysis: 50 randomly sampled outputs, flagged for known issues (hand anatomy, text rendering, consistent object count).

Любая single metric — ложь. Three corroborating metrics + qualitative review — claim.

## Практика

`code/main.py` реализует FID, CLIP-score-like и Elo aggregation на synthetic "feature vectors" (мы используем 4-D vectors как stand-ins for Inception features). Вы увидите:

- FID computation on small N and large N — bias.
- "CLIP score" как cosine similarity between feature pools.
- Elo update rule from synthetic preference stream.

### Шаг 1: FID in four lines

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Шаг 2: CLIP-style cosine-similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Шаг 3: Elo aggregation

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Подводные камни

- **FID at N=1000.** Heuristic unreliable under N=10k. Papers reporting low-N FID are gaming.
- **Comparing FID across resolutions.** Resize Inception до 299×299 меняет feature distribution. Compare at matched resolution only.
- **Reporting one seed.** Run 3 seeds minimum. Report std.
- **CLIP score inflation via negative prompts.** Некоторые pipelines boost CLIP by over-fitting prompt. Check for visual saturation.
- **Elo bias from prompt overlap.** Если обе models видели benchmark prompt during training, Elo meaningless. Use held-out prompt sets.
- **Human eval paid-crowd skew.** Prolific, MTurk annotators skew younger / tech-friendly. Mix with recruited art/design experts.

## Применение

Production eval protocol in 2026:

| Pillar | Minimum | Recommended |
|--------|---------|-------------|
| Sample quality | FID on 10k vs held-out real | + CMMD on 5k + FID on subset per category |
| Prompt adherence | CLIP score on 30k | + HPSv2 + ImageReward + VQA-style question answering |
| Preference | 200 blinded pairs vs baseline | + 2000 paired human + LLM-judge + Chatbot Arena |
| Failure analysis | 50 hand-flagged | 500 hand-flagged + automated safety classifier |

All four pillars in one report = claim. Any one alone = marketing.

## Запуск в продукт

Сохраните `outputs/skill-eval-report.md`. Навык принимает new model checkpoint + baseline и outputs full eval plan: sample sizes, metrics, failure-mode probes, sign-off criteria.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Compare FID at N=100 vs N=1000 on same synthetic distributions. Report bias magnitude.
2. **Средне.** Реализуйте CMMD from synthetic CLIP-style features (см. Jayasumana et al., 2024 for formula). Compare sensitivity to quality differences vs FID.
3. **Сложно.** Replicate HPSv2 setup: take 1000 image-prompt pairs from subset of Pick-a-Pic, fine-tune small CLIP-based scorer on preferences, and measure agreement with held-out set.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | Fréchet distance of Gaussian fits to real vs gen Inception features. |
| CLIP score | "Text-image similarity" | Cosine similarity between CLIP image and text embeddings. |
| CMMD | "FID's replacement" | CLIP-feature MMD; less biased, no Gaussian assumption. |
| IS | "Inception score" | Exp KL(p(y|x) || p(y)); poorly correlates on modern models, retired. |
| HPSv2 / ImageReward / PickScore | "Learned preference proxies" | Small models trained on human preferences; used as automatic judges. |
| Elo | "Chess rating" | Bradley-Terry aggregation of pairwise wins. |
| PartiPrompts | "The benchmark prompt set" | 1,600 Google-curated prompts across 12 categories. |
| FD-DINO | "Self-sup replacement" | FD using DINOv2 features; better for out-of-ImageNet domains. |

## Production note: evaluation is an inference workload too

Running FID on 10k samples means generating 10k images. Для 50-step SDXL base at 1024² on single L4 это ~11 hours of single-request inference. Evaluation budgets are real, and framing exactly offline-inference scenario (maximize throughput, ignore TTFT):

- **Batch hard, forget latency.** Offline eval = static batching at largest size fitting memory. `pipe(...).images` with `num_images_per_prompt=8` on 80GB H100 runs 4-6× faster wall-clock than single-request.
- **Cache the real features.** Inception (FID) или CLIP (CLIP-score, CMMD) feature extraction over real reference set runs *once*, stored as `.npz`. Do not recompute per eval.

Для CI / regression gates: run FID + CLIP score on 500-sample subset per PR (~30 min); run full 10k FID + HPSv2 + Elo nightly.

## Дополнительное чтение

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — статья FID.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — failure-mode survey.
