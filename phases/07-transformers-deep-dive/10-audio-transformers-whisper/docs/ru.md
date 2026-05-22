# Аудиотрансформеры — архитектура Whisper

> Аудио — это изображение частот во времени. Whisper — это ViT, который ест mel-спектрограммы и отвечает речью в виде текста.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 7 · 05 (Full Transformer), Фаза 7 · 08 (Encoder-Decoder), Фаза 7 · 09 (ViT)
**Время:** ~45 минут

## Проблема

До Whisper (OpenAI, Radford et al. 2022) передовое автоматическое распознавание речи (ASR) означало wav2vec 2.0 и HuBERT — самообучаемые извлекатели признаков плюс дообученная голова. Высокое качество, дорогие конвейеры данных, хрупкость при смене домена. Для многоязычного распознавания речи нужны были отдельные модели для каждой языковой семьи.

Whisper сделал три ставки:

1. **Обучаться на всем.** 680 000 часов слабо размеченного аудио, собранного из интернета, на 97 языках. Без чистого академического корпуса. Без меток фонем.
2. **Одна многозадачная модель.** Один декодер, совместно обученный транскрипции, переводу, обнаружению речевой активности, определению языка и расстановке временных меток через task tokens.
3. **Стандартный encoder-decoder transformer.** Энкодер принимает log-mel-спектрограммы. Декодер авторегрессионно производит текстовые токены. Без vocoder, без CTC, без HMM.

Результат: Whisper large-v3 устойчив к акцентам, шуму и языкам, для которых нет чистых размеченных данных. В 2026 году это стандартный речевой front-end для каждого open-source голосового ассистента и большинства коммерческих.

## Концепция

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Шаг 1 — resample + window

Аудио с частотой 16 kHz. Обрезать/дополнить до 30 секунд. Вычислить log-mel-спектрограмму: 80 mel bins, stride 10 ms → ~3 000 кадров × 80 признаков. Это "входное изображение", которое видит Whisper.

### Шаг 2 — convolutional stem

Два слоя Conv1D с kernel 3 и stride 2 уменьшают 3 000 кадров до 1 500. Это вдвое сокращает длину последовательности без большого добавления параметров.

### Шаг 3 — encoder

24-слойный (для large) transformer encoder по 1 500 временным шагам. Синусоидальное позиционное кодирование, self-attention, GELU FFN. Производит скрытые состояния размером 1 500 × 1 280.

### Шаг 4 — decoder

24-слойный transformer decoder. Он авторегрессионно производит токены из BPE-словаря, который является надмножеством GPT-2 с несколькими специальными токенами для аудио.

### Шаг 5 — task tokens

Промпт декодера начинается с управляющих токенов, которые говорят модели, что делать:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

или

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Модель обучалась на этой конвенции. Вы управляете задачей через префикс. Эквивалент instruction-tuning 2026 года, но примененный к речи.

### Шаг 6 — output

Beam search (width 5) с порогом log-prob. Временные метки предсказываются каждые 0.02 секунды аудио, когда токен `<|notimestamps|>` отсутствует.

### Размеры Whisper

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo (2024) сократил декодер с 32 слоев до 4. Декодирование стало в 8× быстрее при регрессии меньше 1 WER point. Именно это ускорение декодирования делает Whisper-turbo стандартом для real-time voice agents в 2026 году.

### Чего Whisper не делает

- Нет diarization (кто говорит). Для этого сочетайте с pyannote.
- Нет нативного real-time streaming — 30-секундное окно фиксировано. Современные обертки (`faster-whisper`, `WhisperX`) добавляют streaming через VAD + overlap.
- Нет long-form context за пределами 30 s без внешнего разбиения на chunks. На практике работает хорошо, потому что человеческой речи редко нужен дальний контекст для транскрипции.

### Ландшафт 2026

| Task | Model | Notes |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine в 4× быстрее на edge |
| Multilingual ASR | Whisper-large-v3 | 97 languages |
| Streaming ASR | faster-whisper + VAD | достижимы цели latency 150 ms |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, но Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens in one transformer |

## Соберите это

См. `code/main.py`. Мы не обучаем Whisper — мы строим конвейер log-mel-спектрограмм + форматтер промпта с task-token. Это те части, с которыми вы действительно работаете в production.

### Шаг 1: синтезировать аудио

Сгенерируйте 1-секундную синусоиду на 440 Hz с частотой дискретизации 16 kHz. 16 000 samples.

### Шаг 2: log-mel spectrogram (упрощенно)

Полная mel-спектрограмма требует FFT. Мы делаем упрощенную версию с разбиением на кадры + энергией по кадру, которая показывает конвейер без необходимости в `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms, hop = 10 ms. Совпадает с windowing в Whisper. Энергия по кадру заменяет mel bins в учебных целях.

### Шаг 3: дополнить до 30 s

Whisper всегда обрабатывает 30-секундные chunks. Дополните (или обрежьте) спектрограмму до 3 000 кадров.

### Шаг 4: построить prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Это вся поверхность управления задачей. 4-токенный префикс.

## Используйте это

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Быстрее, OpenAI-compatible:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Когда выбирать Whisper в 2026:**

- Многоязычный ASR одной моделью.
- Устойчивая транскрипция шумного, разнообразного аудио.
- Research / prototype ASR — самая быстрая стартовая точка.

**Когда выбирать что-то другое:**

- Ultra-low latency streaming on edge — Moonshine обгоняет Whisper при сопоставимом качестве.
- Real-time conversational AI, которому нужно <200 ms — специализированный streaming ASR.
- Speaker diarization — Whisper этого не делает; подключайте pyannote.

## Доведите до поставки

См. `outputs/skill-asr-configurator.md`. Skill выбирает ASR-модель, параметры декодирования и preprocessing pipeline для нового речевого приложения.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Подтвердите, что число кадров для 1-секундного сигнала при 16 kHz с hop 10 ms составляет ~100 кадров. Для 30 секунд: ~3 000 кадров.
2. **Средне.** Постройте полную log-mel-спектрограмму с помощью `numpy.fft`. Проверьте, что 80 mel bins совпадают с `librosa.feature.melspectrogram(n_mels=80)` в пределах численной ошибки.
3. **Сложно.** Реализуйте streaming inference: разбейте аудио на 10 s windows с overlap 2 s, запустите Whisper на каждом chunk, объедините transcripts. Измерьте word-error rate относительно single-pass на 5-минутном фрагменте подкаста.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mel spectrogram | "Audio image" | 2D-представление: frequency bins по одной оси, time frames по другой; log-scaled energy в каждой ячейке. |
| Log-mel | "What Whisper sees" | Mel spectrogram, пропущенная через log; приближает человеческое восприятие громкости. |
| Frame | "One time slice" | Окно samples длиной 25 ms; перекрывается со stride 10 ms. |
| Task token | "Prompt prefix for speech" | Специальные токены вроде `<|transcribe|>` / `<|translate|>` в prompt декодера. |
| Voice activity detection (VAD) | "Find the speech" | Gate, который удаляет тишину перед ASR; резко снижает cost. |
| CTC | "Connectionist Temporal Classification" | Классический ASR loss для обучения без alignment; Whisper его НЕ использует. |
| Whisper-turbo | "Small decoder, full encoder" | large-v3 encoder + 4-layer decoder; декодирование в 8× быстрее. |
| Faster-whisper | "The production wrapper" | Реализация на CTranslate2; int8 quantization; в 4× быстрее reference OpenAI. |

## Дополнительное чтение

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — статья Whisper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — reference code + model weights. Прочитайте `whisper/model.py`, чтобы увидеть Conv1D stem + encoder + decoder сверху донизу примерно в 400 строк.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — здесь находится логика beam-search + task-token, описанная в шагах 5–6; 500 строк, полностью читаемо.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — предшественник; все еще SOTA-признаки в некоторых условиях.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — production wrapper, в 4× быстрее reference.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — edge-friendly ASR 2024 года, Whisper-shaped, но меньше.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — канонический рецепт fine-tuning, включая preprocessor mel-спектрограмм и обработку token-timestamp.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — полная реализация (encoder, decoder, cross-attention, generation), которая повторяет архитектурную диаграмму урока.
