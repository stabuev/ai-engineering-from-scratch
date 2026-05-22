# Text-to-Speech (TTS) — от Tacotron до F5 и Kokoro

> ASR обращает речь в текст; TTS обращает текст в речь. Стек 2026 года состоит из трех частей: text → tokens, tokens → mel, mel → waveform. Для каждой части есть дефолтная модель, помещающаяся в ноутбук.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 02 (спектрограммы и Mel), Фаза 5 · 09 (Seq2Seq), Фаза 7 · 05 (Full Transformer)
**Время:** ~75 минут

## Проблема

У вас есть строка: "Please remind me to water the plants at 6 pm." Нужен 3-секундный аудиоклип, который звучит естественно, имеет правильную просодию (паузы, ударения), произносит "plants" с нужной гласной и работает быстрее 300 ms на CPU для живого голосового ассистента. Еще нужно менять голоса, обрабатывать code-switched input ("remind me at 6 pm, daijoubu?") и не позориться на именах.

Современные TTS-пайплайны выглядят так:

1. **Text frontend.** Нормализовать текст (даты, числа, emails), преобразовать в phonemes или subword tokens, предсказать prosody features.
2. **Acoustic model.** Text → mel spectrogram. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → waveform. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), neural codec vocoders в 2024+.

В 2026 году разделение acoustic + vocoder размывается end-to-end diffusion и flow-matching моделями. Но мысленная модель из трех частей все еще полезна для debugging.

## Концепция

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder emits mel frames. Медленный (AR), нестабилен на длинном тексте. Все еще цитируется как бейзлайн.

**FastSpeech 2 (2020).** Неавторегрессионный. Duration predictor выдает, сколько mel frames получает каждая phoneme. Один проход, в 10× быстрее Tacotron. Теряет немного естественности (monotonic alignment), но поставляется повсюду.

**VITS (2021).** Совместно обучает encoder + flow-based duration + HiFi-GAN vocoder end-to-end с variational inference. Высокое качество, единая модель. Доминировал в open-source TTS 2022–2024. Варианты: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Diffusion transformer на flow matching. Естественная просодия, zero-shot voice cloning по 5 секундам reference audio. Верхушка open-source TTS leaderboards 2026 года. 335M params.

**Kokoro (2024).** Маленький (82M), запускается на CPU, лучший English TTS для real-time use в своем классе. Closed-vocabulary English-only, apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Коммерческий state of the art. ElevenLabs v2.5 с emotion tags ("[whispered]", "[laughing]") и character voices доминирует в производстве аудиокниг в 2026 году.

### Эволюция вокодеров

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

К 2026 году большинство "TTS" моделей end-to-end от текста до waveform; mel-спектрограмма стала внутренним представлением.

### Оценка

- **MOS (Mean Opinion Score).** Шкала 1–5, crowd-sourced. Все еще золотой стандарт; мучительно медленный.
- **CMOS (Comparative MOS).** A-vs-B preference. Более узкие confidence intervals на аннотацию.
- **UTMOS, DNSMOS.** Neural MOS predictors без reference. Используются для leaderboards.
- **CER (Character Error Rate) via ASR.** Прогоните TTS output через Whisper, посчитайте CER против input text. Proxy для intelligibility.
- **SECS (Speaker Embedding Cosine Similarity).** Качество voice cloning.

Числа 2026 года на LibriTTS test-clean:

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## Соберите это

### Шаг 1: фонемизируйте вход

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Фонемы — универсальный мост. Не подавайте сырой текст ниже уровня качества VITS.

### Шаг 2: запустите Kokoro (CPU-дефолт 2026)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Работает offline, один файл, 82M params.

### Шаг 3: запустите F5-TTS с voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Передайте 5-секундный reference clip + его transcript; F5 клонирует prosody и timbre.

### Шаг 4: HiFi-GAN vocoder с нуля

Слишком велик для tutorial script, но форма такая:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Обучение: adversarial (discriminator на short windows) + mel-spectrogram reconstruction loss + feature-matching loss. Коммодитизировано — используйте pretrained checkpoints из `hifi-gan` repo или nvidia-NeMo.

### Шаг 5: полный пайплайн (pseudocode)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) или XTTS v2 (GPU) |
| Voice cloning from 5 s reference | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 или XTTS v2 + fine-tune |
| Low-resource language | Обучить VITS на 5–20 h target-lang data |
| Expressive / emotion tags | ElevenLabs v2.5 или StyleTTS 2 fine-tune |

Open-source лидеры 2026 года: **F5-TTS по качеству, Kokoro по эффективности**. Не тянитесь к Tacotron, если вы не историк.

## Ловушки

- **Нет text normalizer.** "Dr. Smith" читается как "Doctor" или "Drive"? "2026" как "twenty twenty six" или "two zero two six"? Нормализуйте ДО phonemizer.
- **OOV proper nouns.** "Ghumare" → "ghyu-mair"? Поставьте fallback grapheme-to-phoneme model для unknown tokens.
- **Clipping.** Vocoder output редко клипует, но mismatch mel scaling при инференсе может выйти за ±1.0. Всегда `np.clip(wav, -1, 1)`.
- **Несовпадение частоты дискретизации.** Kokoro выдает 24 kHz; downstream pipeline ожидает 16 kHz → resample, иначе получите aliasing.

## Доведите до результата

Сохраните как `outputs/skill-tts-designer.md`. Спроектируйте TTS pipeline для заданного голоса, latency и language target.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он строит phoneme dictionary из toy vocab, оценивает duration per phoneme и печатает fake "mel" schedule.
2. **Средне.** Установите Kokoro, синтезируйте одно и то же предложение голосами `af_bella` и `am_adam`. Сравните durations и субъективное качество.
3. **Сложно.** Запишите 5-секундный reference clip себя. Используйте F5-TTS, чтобы клонировать его. Сообщите SECS между reference и cloned output.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| Phoneme | Единица звука | Абстрактный класс звука; 39 в English (ARPABet). |
| Duration predictor | Сколько длится каждая фонема | Non-AR output модели; целые frames per phoneme. |
| Vocoder | Mel → waveform | Нейросеть, отображающая mel-spec в сырые отсчеты. |
| HiFi-GAN | Стандартный vocoder | GAN-based; доминировал 2020–2024. |
| MOS | Субъективное качество | Mean opinion score 1–5 от human raters. |
| SECS | Метрика voice-clone | Cosine similarity между target и output speaker embedding. |
| F5-TTS | Open-source SOTA 2024 | Flow-matching diffusion; zero-shot cloning. |
| Kokoro | CPU English leader | Модель 82M params, Apache 2.0. |

## Дополнительное чтение

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq-бейзлайн.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — текущий open-source SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — vocoder, который все еще ship в 2026.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — CPU-friendly English TTS 2024 года.
