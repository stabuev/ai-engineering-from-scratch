# Распознавание речи (ASR) — CTC, RNN-T, Attention

> Распознавание речи — это аудиоклассификация на каждом временном шаге, склеенная sequence model, которая знает английский и тишину. CTC, RNN-T и attention — три способа сделать это. Выберите один и поймите почему.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 02 (спектрограммы и Mel), Фаза 5 · 08 (CNN и RNN для текста), Фаза 5 · 10 (Attention)
**Время:** ~45 минут

## Проблема

У вас есть 10-секундный клип 16 kHz. Вы хотите строку: «turn on the kitchen lights». Сложность структурная: аудиофреймы не выравниваются один-к-одному с символами. Слово "okay" может длиться 200 ms или 1200 ms. Тишина пунктуирует высказывание. Одни фонемы длиннее других. Число выходных токенов заранее неизвестно.

Это решают три формулировки:

1. **CTC (Connectionist Temporal Classification).** Выдавать вероятности токенов на каждом фрейме, включая специальный *blank*. При декодировании схлопывать повторы и blank. Неавторегрессионно, быстро. Используется в wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Joint network предсказывает следующий токен по encoder frame и предыдущим токенам. Поддерживает streaming. Используется в on-device ASR Google, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Encoder сжимает аудио в hidden states, decoder делает cross-attention и авторегрессионно генерирует токены. Используется в Whisper, SeamlessM4T.

В 2026 году SOTA WER на LibriSpeech test-clean — 1.4% (Parakeet-TDT-1.1B, NVIDIA) и 1.58% (Whisper-Large-v3-turbo). Различия качества малы; различия деплоя огромны.

## Концепция

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Интуиция CTC.** Пусть encoder выдает `T` распределений по фреймам над `V+1` токенами (V символов + blank). Для целевой строки `y` длины `U < T` засчитывается любое фреймовое выравнивание, которое схлопывается в `y`. CTC loss суммирует по всем таким выравниваниям. Инференс: argmax по фреймам, схлопнуть повторы, удалить blank.

Плюсы: неавторегрессионность, streamable, нулевой lookahead. Минус: *conditional independence assumption* — предсказание каждого фрейма независимо от других, поэтому внутренней языковой модели нет. Исправляют внешней LM через beam search или shallow fusion.

**Интуиция RNN-T.** Добавляет *predictor*, который эмбеддит историю токенов, и *joiner*, который объединяет состояние predictor с encoder frame в совместное распределение над `V+1` (`+1` — null / no-emit). Явно моделирует условные зависимости, которые CTC игнорировал. Streamable, потому что каждый шаг зависит только от прошлых фреймов и прошлых токенов.

Плюсы: streaming + внутренняя LM. Минус: обучение сложнее и прожорливее по памяти (3D loss lattice); RNN-T loss kernels — отдельная категория библиотек.

**Attention encoder-decoder.** Encoder (6-32 transformer layers) по log-mel фреймам. Decoder (6-32 transformer layers) с causal self-attn + cross-attn к выходам encoder генерирует токены авторегрессионно. Ограничений выравнивания нет — attention может смотреть куда угодно в аудио. Не streamable, если не ограничить attention (chunked Whisper-Streaming, 2024).

Плюсы: лучшее качество offline ASR, легко обучать стандартным seq2seq-инструментарием. Минус: авторегрессионная задержка пропорциональна длине вывода; streaming требует инженерии.

### WER: главное число

**Word Error Rate** = `(S + D + I) / N`, где S=substitutions, D=deletions, I=insertions, N=число слов в reference. Это Levenshtein edit distance на уровне слов. Ниже лучше. WER выше 20% обычно непригоден; ниже 5% — человеческий уровень для прочитанной речи. Числа 2026 года:

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Все они основаны на encoder-decoder или RNN-T. Чистые CTC-системы (wav2vec 2.0) находятся примерно на 1.8–2.1% на test-clean.

## Соберите это

### Шаг 1: greedy CTC decode

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Два правила: схлопнуть подряд идущие повторы, удалить blank. Пример: `a a _ _ a b b _ c` → `a a b c`.

### Шаг 2: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

В продакшене используют prefix tree beam search с LM fusion; это концептуальный скелет.

### Шаг 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Шаг 4: инференс через Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Одна строка для самого сильного общего ASR в 2026 году. На 24 GB GPU работает примерно в ~20× realtime.

### Шаг 5: streaming с Parakeet или wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR требует chunked encoder attention и carryover state; используйте библиотеку, которая это поддерживает (NeMo для Parakeet, `transformers` pipeline с `chunk_length_s`).

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Английский, offline, максимум качества | Whisper-large-v3-turbo |
| Многоязычность, устойчивость | SeamlessM4T v2 |
| Streaming, низкая задержка | Parakeet-TDT-1.1B или Riva |
| Edge, mobile, задержка <500 ms | Whisper-Tiny quantized или Moonshine (2024) |
| Long-form | Whisper с VAD-based chunking (WhisperX) |
| Domain-specific (medical, legal) | Дообучить wav2vec 2.0 + domain LM fusion |

## Ловушки, которые все еще попадают в продакшен в 2026 году

- **Нет VAD.** Запуск Whisper на тишине дает галлюцинации ("Thanks for watching!"). Всегда гейтите через VAD.
- **Character vs word vs subword WER.** Сообщайте word-level WER *после* нормализации (lowercase, без punctuation).
- **Дрейф Language ID.** Auto LID Whisper на шумных клипах уводит в Japanese или Welsh; задавайте `language="en"`, когда знаете язык.
- **Длинные клипы без chunking.** У Whisper окно 30 секунд. Используйте `chunk_length_s=30, stride=5` для всего, что длиннее.

## Доведите до результата

Сохраните как `outputs/skill-asr-picker.md`. Выберите модель, стратегию декодирования, chunking и LM fusion для заданной цели деплоя.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он жадно декодирует вручную созданный CTC output и считает WER относительно reference.
2. **Средне.** Реализуйте prefix-tree beam search из шага 2 корректно (с учетом правила слияния blank). Сравните с greedy на синтетическом датасете из 10 примеров.
3. **Сложно.** Используйте `whisper-large-v3-turbo` на [LibriSpeech test-clean](https://www.openslr.org/12). Посчитайте WER на первых 100 utterances. Сравните с опубликованными числами.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| CTC | Loss с blank-token | Маргинализация по всем выравниваниям фрейм-токен; non-AR. |
| RNN-T | Streaming loss | CTC + predictor следующего токена; учитывает порядок слов. |
| Attention enc-dec | Whisper-style | Encoder + decoder с cross-attention; лучшее offline качество. |
| WER | Число, которое вы сообщаете | `(S+D+I)/N` на уровне слов. |
| Blank | Пустота | Специальный токен CTC, означающий "нет эмиссии в этом фрейме". |
| LM fusion | Внешняя языковая модель | Добавление взвешенных LM log-probs при beam search. |
| VAD | Гейт тишины | Детектор речевой активности; обрезает неречь. |

## Дополнительное чтение

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — статья CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — статья RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — каноническая статья 2022 года; расширение v3-turbo в 2024.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — лидер Open ASR Leaderboard 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — живой benchmark по 25+ моделям.
