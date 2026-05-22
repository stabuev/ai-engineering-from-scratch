# Нейронные аудиокодеки — EnCodec, SNAC, Mimi, DAC и semantic-acoustic split

> Генерация аудио 2026 года почти вся работает с tokens. EnCodec, SNAC, Mimi и DAC превращают непрерывные формы волны в дискретные последовательности, которые может предсказывать transformer. Разделение semantic-vs-acoustic tokens — первый codebook как semantic, остальные как acoustic — важнейший архитектурный сдвиг в аудио со времен Transformer.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 6 · 02 (спектрограммы), Фаза 10 · 11 (Quantization), Фаза 5 · 19 (Subword Tokenization)
**Время:** ~60 минут

## Проблема

Language models работают с дискретными tokens. Audio непрерывно. Если нужна LLM-style модель для speech / music — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — сначала нужен **neural audio codec**: обученный encoder, дискретизирующий audio в малый словарь tokens, и соответствующий decoder, реконструирующий waveform.

Появились два семейства:

1. **Reconstruction-first codecs** — EnCodec, DAC. Оптимизируют perceptual audio quality. Tokens «acoustic»: захватывают все, включая speaker identity, timbre, background noise.
2. **Semantic-first codecs** — Mimi (Kyutai), SpeechTokenizer. Заставляют первый codebook кодировать linguistic / phonetic content (часто через distillation from WavLM). Остальные codebooks — acoustic detail.

Инсайт 2024-2026: **codec, оптимизированный только под reconstruction, дает размытую речь при генерации из текста.** LLM над codec tokens приходится учить и language structure, и acoustic structure в одном codebook. Это плохо масштабируется. Разделение — semantic codebook 0, acoustic codebooks 1-N — то, что делает Moshi и Sesame CSM рабочими.

## Концепция

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### Главный трюк: Residual Vector Quantization (RVQ)

Вместо одного огромного codebook (понадобились бы миллионы codes) современные audio codecs используют **RVQ**: каскад маленьких codebooks. Первый квантует encoder output; второй квантует residual; и так далее. Каждый codebook — 1024 codes. 8 codebooks = effective vocabulary 1024^8 = 10^24.

При инференсе decoder суммирует выбранные codes по frame, чтобы реконструировать сигнал.

### Четыре важных кодека 2026 года

**EnCodec (Meta, 2022).** Бейзлайн. Encoder-decoder поверх waveform, RVQ bottleneck. 24 kHz, возможно 32 codebooks, default 4 codebooks @ 1.5 kbps. Архитектура `1D conv + transformer + 1D conv`. Используется MusicGen.

**DAC (Descript, 2023).** RVQ с L2-normalized codebooks, periodic activation functions, улучшенными losses. Самая высокая reconstruction fidelity среди open codecs — иногда неотличим от original speech с 12 codebooks. 44.1 kHz full-band.

**SNAC (Hubert Siuzdak, 2024).** Multi-scale RVQ — coarse codebooks работают на меньшем frame rate, чем fine. Иерархически моделирует audio: грубый "sketch" на ~12 Hz плюс detail на 50 Hz. Используется Orpheus-3B, потому что hierarchy хорошо ложится на LM generation.

**Mimi (Kyutai, 2024).** Game-changer 2026 года. Frame rate 12.5 Hz, 8 codebooks @ 4.4 kbps. Codebook 0 **distilled from WavLM** — учится предсказывать speech-content features WavLM. Codebooks 1-7 — acoustic residuals. Это питает Moshi (Lesson 15) и Sesame CSM.

### Frame rates важны для language modeling

Ниже frame rate = короче sequence = быстрее LM.

| Codec | Frame rate | 1 s = N frames | Для чего подходит |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

При 12.5 Hz 10-секундное utterance — всего 125 codec frames, transformer легко их предсказывает.

### Semantic vs acoustic tokens

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token (codebook 0 in Mimi).** Кодирует, что было сказано: phonemes, words, content. Distilled from WavLM через auxiliary prediction loss.
- **Acoustic tokens (codebooks 1-7).** Кодируют timbre, speaker identity, prosody, background noise, fine detail.

AR LM сначала предсказывает semantic token (conditioned on text), затем acoustic tokens (conditioned on semantic + speaker reference). Эта факторизация объясняет, почему modern TTS умеет zero-shot-clone voices: semantic model отвечает за content, acoustic model — за timbre.

### Reconstruction quality 2026 (bits per sec, ниже bitrate — лучше)

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Традиционные codecs вроде Opus все еще выигрывают по perceptual quality на бит. Neural codecs выигрывают за счет **discrete tokens** и **качества generative models**.

## Соберите это

### Шаг 1: encode с EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` при 6 kbps. Каждый code — 0-1023 (10-bit).

### Шаг 2: decode и измерьте reconstruction

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Шаг 3: semantic-acoustic split (Mimi-style)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 выровнен с WavLM. Можно обучить text-to-semantic transformer — словарь меньше, чем direct-to-audio. Затем отдельный acoustic-to-waveform decoder conditions on speaker reference.

### Шаг 4: почему AR LM поверх codec tokens работает

Для 10 s speech clip при Mimi 12.5 Hz × 8 codebooks:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens — тривиальный context для transformer. Transformer на 256M parameters может генерировать 10 секунд речи за milliseconds на modern GPU.

## Используйте это

Сопоставьте задачу с codec:

| Задача | Codec |
|------|-------|
| Общая генерация музыки | EnCodec-24k |
| Максимально качественная reconstruction | DAC-44.1k |
| AR LM поверх speech (TTS) | SNAC или Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound-effect library with text | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

Правило: **если строите generative model, начинайте с Mimi или SNAC. Если строите compression pipeline, используйте Opus.**

## Ловушки

- **Too many codebooks.** Codebooks повышают fidelity линейно, но и LM sequence length линейно. Остановитесь на 8-12.
- **Frame-rate mismatch.** Training LM на 12.5 Hz Mimi, затем fine-tuning на 50 Hz EnCodec silently fails.
- **Assuming all codebooks equal.** В Mimi codebook 0 несет content; потеря его уничтожает intelligibility. Потеря codebook 7 почти незаметна.
- **Using reconstruction quality as the only metric.** Codec может отлично реконструировать, но быть бесполезным для LM-based generation, если semantic structure плохая.

## Доведите до результата

Сохраните как `outputs/skill-codec-picker.md`. Выберите codec для заданной generative или compression task.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он реализует toy scalar + residual quantizer и измеряет reconstruction error при добавлении codebooks.
2. **Средне.** Установите `encodec` и сравните 1, 4, 8, 32 codebooks на held-out speech clip. Постройте PESQ или MSE vs bitrate.
3. **Сложно.** Загрузите Mimi. Закодируйте clip. Замените codebook 0 случайными integers; декодируйте. Затем так же замените codebook 7. Сравните corruptions — codebook 0 должен уничтожить intelligibility; codebook 7 почти ничего не изменить.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | Каскад маленьких codebooks; каждый квантует предыдущий residual. |
| Frame rate | Codec speed | Сколько token-frames в секунду. Ниже = быстрее LM. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook distilled from SSL features; кодирует content. |
| Acoustic codebooks | Все остальное | Timbre, prosody, noise, fine detail. |
| PESQ / ViSQOL | Perceptual quality | Objective metrics, correlating with MOS. |
| EnCodec | Meta codec | RVQ baseline; используется MusicGen. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; powers Moshi. |

## Дополнительное чтение

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ baseline.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — highest-fidelity open.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — multi-scale RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — semantic-acoustic split, WavLM distillation.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — two-stage semantic/acoustic paradigm.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — original streamable RVQ codec.
