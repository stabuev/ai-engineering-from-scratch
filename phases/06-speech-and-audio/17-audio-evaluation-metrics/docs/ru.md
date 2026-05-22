# Оценка аудио — WER, MOS, UTMOS, MMAU, FAD и открытые лидерборды

> Нельзя выпускать то, что нельзя измерить. Этот урок называет метрики 2026 года для каждой аудиозадачи: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-language (MMAU, LongAudioBench), music (FAD, CLAP) и speaker (EER). Плюс лидерборды, где сравниваться.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 6 · 04, 06, 07, 09, 10; Фаза 2 · 09 (Model Evaluation)
**Время:** ~60 минут

## Проблема

У каждой аудиозадачи несколько метрик, и каждая измеряет свою ось. Неверная метрика — способ выпустить модель, которая отлично выглядит на dashboard и ужасно работает в production. Канонический список 2026 года:

| Задача | Основная | Дополнительные |
|------|---------|-----------|
| ASR | WER | CER · RTFx · latency первого токена |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR в рабочей точке |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · recall по классам |
| Music generation | FAD | CLAP · MOS слушательской панели |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## Концепция

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### Метрики ASR

**WER (Word Error Rate).** `(S + D + I) / N`. Перед подсчетом приведите к нижнему регистру, удалите пунктуацию и нормализуйте числа. Используйте `jiwer` или `whisper_normalizer` OpenAI. &lt; 5% = уровень человека для read speech.

**CER (Character Error Rate).** Та же формула на уровне символов. Используется для tone languages (Mandarin, Cantonese), где сегментация на слова неоднозначна.

**RTFx (inverse real-time factor).** Сколько секунд аудио обрабатывается за одну секунду реального времени. Больше — лучше. Parakeet-TDT достигает 3380×. Whisper-large-v3 — ~30×.

**First-token latency.** Реальное время от аудиовхода до первого токена transcript. Критично для streaming. Deepgram Nova-3: ~150 ms.

### Метрики TTS

**MOS (Mean Opinion Score).** Человеческая оценка 1-5. Золотой стандарт, но медленный. Собирайте 20+ слушателей на sample, 100+ samples на модель.

**UTMOS (2022-2026).** Обученный MOS predictor. Коррелирует с человеческим MOS примерно на 0.9 на стандартных benchmarks. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** Для voice cloning. Cosine similarity ECAPA embeddings между reference и cloned output. &gt; 0.75 = узнаваемый клон.

**WER-on-ASR-round-trip.** Запустите Whisper по TTS output, посчитайте WER против входного текста. Ловит регрессии разборчивости. SOTA 2026: &lt; 2% CER.

**TTFA (time-to-first-audio).** Задержка в реальном времени. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### Специфично для voice cloning

**SECS + MOS + CER** как тройка. Клон с высоким SECS, но низким MOS — тембр правильный, но звучание неестественное; наоборот — естественный голос, но не тот speaker.

### Speaker verification

**EER (Equal Error Rate).** Threshold, где False Accept Rate равен False Reject Rate. ECAPA на VoxCeleb1-O: 0.87%.

**minDCF (min Detection Cost).** Взвешенная стоимость в выбранной рабочей точке (часто FAR=0.01). Более релевантна для продакшена, чем EER.

### Diarization

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Пропущенная речь + ложная речевая активность + перепутанные говорящие как доля. AMI meetings: реалистичный DER ~10-20%. pyannote 3.1 + коммерческий Precision-2: &lt;10% DER на хорошо записанном аудио.

**JER (Jaccard Error Rate).** Альтернатива DER, устойчивая к смещению из-за коротких сегментов.

### Audio classification

Multi-label: **mAP (mean Average Precision)** по всем классам. AudioSet: 0.548 mAP для BEATs-iter3.

Взаимоисключающие классы: **top-1, top-5 accuracy**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

Несбалансированные классы: **macro F1** + **recall по классам**. Отчитывайтесь по классам: агрегированная accuracy скрывает, какие классы ломаются.

### Music generation

**FAD (Fréchet Audio Distance).** Расстояние между распределениями VGGish embeddings для настоящего и сгенерированного аудио. MusicGen-small на MusicCaps: 4.5. MusicLM: 4.0. Ниже — лучше.

**CLAP Score.** Оценка соответствия text-audio через CLAP embeddings. &gt; 0.3 = разумное соответствие.

**Listening panel MOS.** Все еще решающее слово для consumer-grade music. Suno v5 ELO 1293 на TTS Arena (из парных человеческих предпочтений).

### Audio-language benchmarks

**MMAU (Massive Multi-Audio Understanding).** 10k пар audio-QA.

**MMAU-Pro.** 1800 сложных items, четыре категории: speech / sound / music / multi-audio. Случайный шанс 25% при 4 вариантах. Gemini 2.5 Pro overall ~60%; multi-audio ~22% по всем моделям.

**LongAudioBench.** Многоминутные клипы с семантическими запросами. Audio Flamingo Next обходит Gemini 2.5 Pro.

**AudioCaps / Clotho.** Captioning benchmarks. SPICE, CIDEr, FENSE metrics.

### Streaming speech-to-speech

**Latency P50 / P95 / P99.** Реальное время от конца речи пользователя до первого слышимого ответа. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**WER / MOS** на выходе.

**Barge-in responsiveness.** Время от перебивания пользователем до замолкания ассистента. Цель &lt; 150 ms.

### Leaderboards 2026

| Лидерборд | Треки | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## Соберите это

### Шаг 1: WER с нормализацией

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Шаг 2: TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Шаг 3: SECS для voice cloning

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Шаг 4: FAD для генерации музыки

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Шаг 5: EER для speaker verification (тот же код, что в уроке 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Используйте это

Свяжите каждый deploy с фиксированным eval harness, который запускается при каждом обновлении модели. Три главных правила:

1. **Normalize before scoring.** Нижний регистр, удаление пунктуации, разворачивание чисел. Сообщайте правило нормализации.
2. **Report distributions, not averages.** P50/P95/P99 для latency. Recall по классам для classification. Разбивка по категориям для MMAU.
3. **Run one canonical public benchmark.** Даже если production data отличается, Open ASR / TTS Arena / MMAU позволяют reviewers сравнивать на одной основе.

## Ловушки

- **UTMOS extrapolation.** Обучен на чистой речи в стиле VCTK; плохо оценивает noisy / cloned / emotional audio.
- **MOS panel bias.** 20 работников Amazon Mechanical Turk ≠ 20 целевых пользователей. Для high-stakes платите за domain panel.
- **FAD depends on reference set.** Сравнивайте с одним и тем же reference distribution для всех models.
- **Aggregate WER.** 5% WER overall может скрывать 30% WER на accented speech. Отчитывайтесь по демографическим срезам.
- **Public benchmark saturation.** Frontier models близки к потолку на стандартных benchmarks. Соберите внутренний held-out set, отражающий ваш traffic.

## Доведите до результата

Сохраните как `outputs/skill-audio-evaluator.md`. Выберите метрики, benchmarks и reporting format для любого release аудиомодели.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Посчитайте WER / CER / EER / SECS / FAD-ish / MMAU-ish на toy inputs.
2. **Средне.** Соберите TTS round-trip WER harness. Прогоните output Kokoro или F5-TTS через Whisper. Посчитайте WER по 50 prompts. Пометьте prompts с WER &gt; 10%.
3. **Сложно.** Оцените LALM из Lesson 10 на поднаборах MMAU-Pro speech + multi-audio (по 50 items). Сообщите accuracy по категориям и сравните с опубликованным числом.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| WER | ASR score | `(S+D+I)/N` на уровне слов после нормализации. |
| CER | Character WER | Для tone languages или char-level systems. |
| MOS | Human opinion | Оценка 1-5; 20+ слушателей × 100 samples. |
| UTMOS | ML MOS predictor | Обученная модель; коррелирует с human MOS примерно на 0.9. |
| SECS | Voice-clone similarity | ECAPA cosine между reference и clone. |
| EER | Speaker verif score | Threshold, где FAR = FRR. |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
| FAD | Music-gen quality | Fréchet distance по VGGish embeddings. |
| RTFx | Throughput | Секунды аудио на секунду реального времени. |

## Дополнительное чтение

- [jiwer](https://github.com/jitsi/jiwer) — библиотека WER/CER с утилитами нормализации.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — обученный MOS predictor.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — стандарт music-gen.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — live rankings 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — TTS leaderboard на человеческих голосованиях.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — leaderboard reasoning для LALM.
- [HEAR benchmark](https://hearbenchmark.com/) — benchmarks audio SSL.
