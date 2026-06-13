# Генерация видео

> Изображение — это 2-D tensor. Видео — 3-D tensor. Теория та же; compute сложнее в 10-100x. Sora от OpenAI (Feb 2024) доказала, что это возможно. К 2026 году Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0 и WAN 2.2 поставляют production video from text at 1080p, а open-weights stack (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) отстает примерно на 12 месяцев.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 07 (Latent Diffusion), Фаза 7 · 09 (ViT), Фаза 8 · 06 (DDPM)
**Время:** ~45 минут

## Цели обучения

- Разбивать видео на пространственно-временные патчи-токены и прогонять DiT-денойзер по всей последовательности.
- Добавлять позиционные эмбеддинги по кадрам и проверять временную согласованность.
- Называть рычаги продакшен-видеомоделей (видео-VAE, факторизованное внимание, image-to-video, re-captioning).

## Проблема

10-секундное 1080p video at 24fps — это 240 frames по 1920×1080×3 pixels. Это ~1.5 GB raw data на clip. Pixel-space diffusion infeasible. Нужно:

1. **Spatiotemporal compression.** VAE, который encodes videos, а не frames, в sequence of spatial-temporal patches.
2. **Temporal coherence.** Frames должны сохранять content, lighting и object identity в течение секунд. Net должна model motion.
3. **Compute budget.** Video training в 10-100x дороже image при том же model size.
4. **Conditioning.** Text, image (first-frame), audio или another video. Большинство production models принимают все четыре.

Архитектура, решившая это, — **Diffusion Transformer (DiT)** на spatiotemporal patches, обученный на огромных (prompt, caption, video) datasets. Та же diffusion loss, что в Lesson 06.

## Концепция

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

Encode video с 3D VAE (learned spatiotemporal compression). Latent имеет форму `[T_latent, H_latent, W_latent, C_latent]`. Разбейте на patches размера `[t_p, h_p, w_p]`. Для Sora-style models `t_p = 1` (per-frame patches) или `t_p = 2` (каждые два frames). 10-секундное 1080p video сжимается примерно в ~20,000-100,000 patches.

### Spatiotemporal DiT

Transformer обрабатывает flat sequence of patches. У каждого patch есть 3D positional embedding (time + y + x). Attention обычно factorized:

- **Spatial attention** внутри patches каждого frame.
- **Temporal attention** across frames в той же spatial location.
- **Full 3D attention** дороже в 16-100x; используется только на low resolution или в research.

### Text conditioning

Cross-attention с large text encoder (T5-XXL для Sora, CogVideoX-5B использует T5-XXL). Long prompts важны — training set Sora имел GPT-generated dense re-captions в среднем 200 tokens на clip.

### Training

Standard diffusion loss (ε или v prediction) по spatiotemporal latents. Data: web video + ~100M curated clips + synthetic text captions. Compute: 10,000+ GPU hours даже для small research run; Sora-scale — 100,000+.

## Производственный ландшафт 2026

| Model | Date | Max duration | Max res | Open weights? | Notable |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | First model to show world simulator properties at scale |
| Sora Turbo | 2024-12 | 20s | 1080p | No | Production Sora at 5x faster inference |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | Highest quality + physics in 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | No | Native audio and stronger camera control |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | Best human motion in 2025 Q1 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | Professional video tools on top |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | Strongest character consistency |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | First open 5B-scale video |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | Open SOTA late 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | Most permissively licensed |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | Strongest open model mid-2025 |

Open weights закрывают gap быстрее, чем в image space: HunyuanVideo + WAN 2.2 LoRAs уже питают большинство open-source workflows к mid-2026.

## Практика

`code/main.py` симулирует core spatiotemporal DiT idea: patchify small synthetic video, добавить per-patch position embedding и denoise whole sequence с transformer-style attention over patches. Без numpy; pure Python. Мы показываем, что temporal coherence возникает даже в 1-D, когда adjacent-frame patches делят denoiser и position embeddings.

### Шаг 1: patchify a synthetic 1-D "video"

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Шаг 2: position embedding per frame

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Шаг 3: denoiser sees the whole sequence

Вместо denoising each frame independently наша tiny net concatenates all frame values + their position embeddings и predicts noise for all frames jointly.

### Шаг 4: temporal coherence test

После training sample video. Измерьте frame-to-frame delta. Если model learned temporal structure, deltas остаются меньше, чем при sampling each frame independently.

## Подводные камни

- **Independent per-frame sampling = flicker.** Если запускать image diffusion на каждом frame отдельно, output flickers, потому что noise каждого frame independent. Video diffusion исправляет это, связывая frames через attention или shared noise.
- **Naive 3D attention = OOM.** Full 3D attention на 10-second 1080p latent — сотни миллиардов operations. Factorize into spatial + temporal.
- **Data captioning matters more than size.** Главный upgrade Sora над prior work — training на ~10x more detailed captions (GPT-4 re-labelled clips). Technical report OpenAI говорит об этом прямо.
- **First-frame conditioning.** Большинство production models также принимают image as first frame. Это "image-to-video" mode; training включает этот variant.
- **Physics drift.** Long clips (>10s) накапливают subtle inconsistencies. Sliding-window generation + keyframe anchoring помогает.

## Применение

| Use case | 2026 pick |
|----------|-----------|
| Highest-quality text-to-video, hosted | Veo 3 or Sora |
| Camera-controlled cinematic | Runway Gen-3 with motion brushes |
| Character consistency across clips | Pika 2.0 or Kling 2.1 |
| Open weights, fast fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V, or Runway |
| Audio-to-video lip sync | Veo 3 (native audio) or a dedicated lip-sync model |
| Video editing | Runway Act-Two, Kling Motion Brush, Flux-Kontext (still-frame) |

Cost per second of video at quality parity упала в 20x между 2024 и 2026.

## Запуск в продукт

Сохраните `outputs/skill-video-brief.md`. Навык принимает video brief (duration, aspect ratio, style, camera plan, subject consistency, audio) и выдает: model + hosting, prompt scaffolding (camera language, subject description, motion descriptors), seed + reproducibility protocol и frame-level QA checklist.

## Упражнения

1. **Легко.** В `code/main.py` сравните frame-to-frame delta для (a) independent per-frame sampling, (b) joint sequence sampling. Сообщите mean и variance deltas.
2. **Средне.** Добавьте first-frame condition: pin frame 0 to a given value и sample rest. Измерьте, как pinned value propagates.
3. **Сложно.** Используйте HuggingFace diffusers, чтобы запустить CogVideoX-2B на local GPU. Замерьте 20 inference steps at 720p for a 6-second clip. Profile spatiotemporal attention, чтобы найти bottleneck.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Video VAE | "3-D VAE" | Encoder, который сжимает `(T, H, W, C)` → spatiotemporal latent. |
| Patches | "The tokens" | Fixed-size 3-D blocks of latent; input to DiT. |
| Factorized attention | "Spatial + temporal" | Attention over space, then over time; skip full 3-D attention. |
| Image-to-video (I2V) | "Animate this photo" | Model takes image + text, outputs video that starts from it. |
| Keyframe conditioning | "Anchor frames" | Pin specific frames to control video arc. |
| Motion brush | "Directional hint" | UI input, где user paints motion vectors onto image. |
| Re-captioning | "Dense captions" | Использование LLM для re-label training clips detailed prompts. |
| Flicker | "Temporal artifact" | Frame-to-frame inconsistency; fixed with coupled denoising. |

## Production note: video latents — проблема memory bandwidth

10-second 1080p clip at 24 fps — это 240 frames × 1920 × 1080 × 3 ≈ 1.5 GB raw pixels. После 4× video VAE compression (`2 × spatial × 2 × temporal`) latent ~100 MB per request. Прогоните это через spatiotemporal DiT for 30 steps at batch 1, и вы двигаете ~3 GB/step через HBM — bottleneck это memory bandwidth, не FLOPs.

Три production knobs, все напрямую из production-inference literature:

- **TP across the DiT.** Text-to-video models routinely ≥10B params. TP=4 across 4 H100s — standard; PP=2 × TP=2 для 405B-class models. Latency per step падает почти linearly with TP до all-reduce wall.
- **Frame batching = continuous batching.** At generation time video conceptually a batch of frames linked by attention. Continuous batching (in-flight scheduling) applies: start rendering frame `t+1` while frame `t-1` is being returned, if architecture allows sliding-window generation.
- **Clip-level prefill cache.** Для image-to-video first-frame conditioning аналогична LLM prompt prefill: compute once, reuse across temporal decoder passes. Это фактически KV-cache for video.

## Дополнительное чтение

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora technical report.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — open SOTA mid-2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — seminal video diffusion paper.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — предок Stable Video Diffusion.
