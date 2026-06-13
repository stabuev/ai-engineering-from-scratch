# Генерация аудио

> Audio — это 1-D signal at 16-48 kHz. Пятисекундный clip — 80-240k samples. Ни один transformer не attends to такой sequence напрямую. Решение для каждой production audio model в 2026 году одно: neural codec (Encodec, SoundStream, DAC) сжимает audio в discrete tokens at 50-75 Hz, а transformer или diffusion model генерирует tokens.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 02 (Audio Features), Фаза 6 · 04 (ASR), Фаза 8 · 06 (DDPM)
**Время:** ~45 минут

## Цели обучения

- Объяснять, почему генерация звука работает на токенах нейрокодека (RVQ), а не на сырых сэмплах.
- Обучать маленький предсказатель токенов и сэмплировать звук условно.
- Противопоставлять две продакшен-парадигмы — авторегрессионное предсказание токенов и flow matching.

## Проблема

Три задачи audio generation:

1. **Text-to-speech.** Дан text, нужно произвести speech. Clean speech — narrow-band и имеет сильную phonetic structure; хорошо решается transformer-over-tokens. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Music generation.** Дан prompt (text, melody, chord progression, genre), нужно произвести music. Распределение гораздо шире. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Audio effects / sound design.** Дан prompt, нужно произвести ambient sound или Foley. AudioGen, AudioLDM 2, Stable Audio Open.

Все три работают на одной основе: neural audio codec + token-AR или diffusion generator.

## Концепция

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Neural audio codecs

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Convolutional encoder сжимает waveform в per-timestep vector; residual vector quantization (RVQ) превращает каждый vector в cascade of K codebook indices. Decoder обращает процесс. 24 kHz audio at 2 kbps using 8 RVQ codebooks at 75 Hz = 600 tokens/sec.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Две generative paradigms сверху

**Token-autoregressive.** Flatten RVQ tokens в sequence, запустить decoder-only transformer. MusicGen использует "delayed parallel", чтобы выдавать K codebook streams параллельно с per-stream offsets. VALL-E генерирует speech tokens из text prompt + 3-second voice sample.

**Latent diffusion.** Pack codec tokens как continuous latents или model them with categorical diffusion. Stable Audio 2.5 использует flow matching на continuous audio latents. AudioLDM 2 использует text-to-mel-to-audio diffusion.

Тренд 2024-2026: flow matching выигрывает для music (faster inference, cleaner samples), а token-AR все еще доминирует speech, потому что naturally causal и хорошо streams.

## Производственный ландшафт

| System | Task | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

## Практика

`code/main.py` симулирует core idea: train tiny next-token transformer на synthetic "audio token" sequences из двух разных "styles" (alternating low and high tokens for style A, monotonic ramp for style B). Condition on style and sample.

### Шаг 1: synthetic audio tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Шаг 2: train a tiny token predictor

Bigram-style predictor conditioned on style. Суть паттерна: codec tokens → cross-entropy training → autoregressive sampling.

### Шаг 3: sample conditionally

Имея style token и starting token, sample следующий token из predicted distribution. Продолжать 20-40 tokens.

## Подводные камни

- **Codec quality caps output quality.** Если codec не может faithfully represent sound, качество generator не поможет. DAC — текущий open best.
- **RVQ error accumulation.** Каждый RVQ layer моделирует residual предыдущего. Ошибки layer 1 propagate. Sampling with temperature 0 on higher layers помогает.
- **Musical structure.** 30 seconds of tokens — это 20k+ tokens at 75 Hz. Сложно для transformers. MusicGen использует sliding window + prompt continuation; Stable Audio — shorter clips + crossfading.
- **Artifacts at boundaries.** Crossfading между generated clips требует аккуратного overlap-add.
- **Clean-data appetite.** Music generators нужны десятки тысяч часов licensed music. Suno / Udio RIAA lawsuit (2024) вынес это на поверхность.
- **Voice cloning ethics.** 3-second sample plus text prompt достаточно, чтобы VALL-E / XTTS / ElevenLabs клонировали voice. Каждой production model нужны abuse detection + opt-out lists.

## Применение

| Task | 2026 stack |
|------|------------|
| Commercial TTS | ElevenLabs, OpenAI TTS, or Azure Neural |
| Voice cloning (consent-verified) | XTTS v2 (open) or ElevenLabs Pro |
| Background music, fast | Stable Audio 2.5 API, Suno, or Udio |
| Music with lyrics | Suno v4 or Udio v1.5 |
| Sound effects / Foley | AudioCraft 2, ElevenLabs SFX, or Stable Audio Open |
| Real-time voice agent | GPT-4o realtime or Gemini Live |
| Open-weights music research | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / translation | HeyGen, ElevenLabs Dubbing |

## Запуск в продукт

Сохраните `outputs/skill-audio-brief.md`. Навык принимает audio brief (task, duration, style, voice, license) и выдает: model + hosting, prompt format (genre tags, style descriptors, structural markers), codec + generator + vocoder chain, seed protocol и eval plan (MOS / CLAP score / CER for TTS / user A/B).

## Упражнения

1. **Легко.** Запустите `code/main.py` и явно задайте style. Проверьте, что generated sequences match style pattern.
2. **Средне.** Добавьте delayed parallel decoding: симулируйте 2 streams of tokens, которые должны оставаться offset by 1 step. Обучите joint predictor.
3. **Сложно.** Используйте HuggingFace transformers для запуска MusicGen-small локально. Generate 10-second clip с тремя different prompts; A/B for style adherence.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Codec | "Neural compression" | Encoder / decoder для audio; typical output — 50-75 Hz tokens. |
| RVQ | "Residual VQ" | Cascade of K quantizers; каждый моделирует residual предыдущего. |
| Token | "One codec symbol" | Discrete index into codebook; 1024 или 2048 typical. |
| Delayed parallel | "Offset codebooks" | Emit K token streams со staggered offsets, чтобы сократить sequence length. |
| Flow matching | "The 2024 win for audio" | Straighter-path alternative to diffusion; faster sampling. |
| Voice prompt | "3-second sample" | Speaker embedding или token prefix, который steers cloned voice. |
| Mel spectrogram | "The visual" | Log-magnitude perceptual spectrogram; используется многими TTS systems. |
| Vocoder | "Mel to wave" | Neural component, converting mel spectrograms back to audio. |

## Production note: audio — streaming problem

Audio — единственная output modality, которую пользователи ожидают получать *по мере generation*, а не all-at-once. В production terms это означает, что важен TPOT (Time Per Output Token), потому что target throughput — скорость listening, а не reading. Для 16kHz audio tokenized at ~75 tokens/second (Encodec) server должен генерировать ≥75 tokens/sec per user, чтобы playback был smooth.

Два architectural consequences:

- **Flow-matching audio models cannot stream trivially.** Stable Audio 2.5 и AudioCraft 2 render fixed clip length in one pass. Для stream нужно chunk clip и overlap boundaries — sliding-window diffusion — добавляя 100-300ms latency overhead против codec AR model.

Если product — "live voice chat" или "real-time music continuation", выбирайте codec AR path. Если это "render a 30-second clip on submit", flow-matching выигрывает по quality и total latency.

## Дополнительное чтение

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec standard.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — первый широко используемый neural audio codec.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — text-to-music 2025 with flow matching.
