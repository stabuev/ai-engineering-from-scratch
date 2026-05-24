# Аудио-языковые модели: дуга от Whisper до Audio Flamingo 3

> Whisper (Radford et al., декабрь 2022) закрепил распознавание речи — 680k часов слабо-размеченной многоязычной речи, простой encoder-decoder transformer, benchmark, из-за которого каждый последующий ASR-релиз ссылался на него. Но распознавание — не рассуждение. Вопросы "what instruments are in this recording", "what emotion is the speaker expressing" или "what happened at minute 3" требуют понимания аудио, а не транскрипции. Qwen-Audio, SALMONN, LTU и NVIDIA Audio Flamingo 3 (AF3, июль 2025) постепенно построили этот стек: сохранить encoder класса Whisper, добавить Q-formers, обучить на audio-text instruction data, добавить chain-of-thought reasoning. Этот урок проходит по всей дуге.

**Тип:** Практика
**Языки:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Предварительные требования:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Время:** ~180 минут

## Цели обучения

- Вычислить log-Mel spectrogram из waveform: windowing, FFT, filter banks, log transform.
- Сравнить варианты encoder: Whisper encoder, BEATs, AF-Whisper hybrid. Когда каждый выигрывает.
- Построить audio Q-former: N learnable queries, cross-attending к spectrogram patches.
- Объяснить cascaded (Whisper-then-LLM) и end-to-end audio-LLM training: почему end-to-end лучше масштабируется для reasoning.

## Проблема

Распознавание речи было решено Whisper. OCR-of-audio стал товарной функцией. Но "commodity" заканчивается на транскрипции. Если модель не может рассуждать о том, что услышала — timing, speakers, emotion, music structure, environmental sounds, — одной транскрипции недостаточно для продуктовых функций.

Три очевидных маршрута:

1. Cascade: Whisper транскрибирует, LLM рассуждает по transcript. Работает для сценариев чистой речи. Проваливается на музыке, environmental audio, overlap нескольких говорящих, emotion.

2. End-to-end audio-LLM: audio encoder подает audio tokens напрямую в LLM, минуя транскрипцию. Сохраняет акустическую информацию (emotion, speaker, environment). Требует новых обучающих данных.

3. Hybrid: audio encoder + text decoder, который может и транскрибировать, и рассуждать. Qwen-Audio и Audio Flamingo выбирают этот маршрут.

## Концепция

### Log-Mel spectrogram: the input feature

Каждый audio encoder начинается с одного и того же признака: log-Mel spectrogram.

1. Resample to 16 kHz.
2. Short-time Fourier transform with 25ms windows, 10ms hop.
3. Take magnitude of the FFT result.
4. Apply Mel filter banks (typically 80 filters log-spaced 0-8000 Hz) to warp to perceptual frequency.
5. Log compress (log(1 + x)) for dynamic range.

Результат: 2D array формы (T, 80), где T — число time frames. Для 30-second clip при 100 Hz frame rate: (3000, 80).

### Whisper's encoder

Encoder Whisper — это 12-layer ViT-style transformer, который обрабатывает log-Mel spectrogram как последовательность time frames. Выход: один hidden-state vector на time frame.

Для ASR decoder Whisper — это cross-attention transformer, который генерирует text tokens, conditioned on encoder output. Стандартный encoder-decoder.

Для ALMs (audio-LLMs) нужен encoder output как вход в другую LLM. Паттерн: Whisper encoder frozen, Q-former trainable, LLM frozen или tuned.

### BEATs and audio-specific encoders

Whisper обучался на данных с доминированием речи. Он слабее для музыки и environmental audio.

BEATs (Chen et al., 2022) — self-supervised transformer, обученный на AudioSet. Он лучше Whisper при том же количестве параметров захватывает музыку и environmental sounds.

AF-Whisper (гибрид Audio Flamingo 3): concat Whisper + BEATs features как audio input. Whisper несет linguistic signal, BEATs несет acoustic signal.

### Audio Q-former

Тот же паттерн, что visual Q-former в BLIP-2. Фиксированное число learnable queries (часто 32 или 64) выполняет cross-attend по output frames аудио-encoder. Queries становятся audio tokens, потребляемыми LLM.

Training alignment stage: только Q-former, contrastive + captioning losses на audio-text pairs (AudioCaps, Clotho). Instruction stage: end-to-end, unfreeze LLM, train on instruction data.

### The arc — SALMONN, Qwen-Audio, AF3

SALMONN (Tang et al., 2023): Whisper + BEATs + Q-former + LLaMA. Первая открытая audio-LLM с серьезной способностью к reasoning. На benchmark MMAU показывает ~0.55 composite.

Qwen-Audio (Chu et al., 2023): похожая архитектура, обучена на более богатом датасете, настроена для multi-turn dialogue. MMAU ~0.60.

LTU — Listen, Think, Understand (Gong et al., 2023): явные reasoning data, фокус на chain-of-thought поверх аудиоклипов. Меньше, но сфокусированнее.

Audio Flamingo 3 (Goel et al., июль 2025): текущая открытая SOTA. 8B LLM backbone (Qwen2 7B), Whisper-large encoder concat BEATs, 64-query Q-former, обучение на 1M+ audio-text instruction pairs. MMAU 0.72, на некоторых sub-tasks соответствует proprietary frontier.

AF3 также вводит on-demand chain-of-thought для аудио: модель может опционально выдавать thinking tokens ("let me identify the instruments first: ...") перед финальным ответом. Accuracy на сложных reasoning tasks повышается на 3-5 points, когда thinking enabled.

### Cascaded vs end-to-end

Cascaded pipeline:

1. Whisper transcribes audio → text.
2. LLM reasons over text.

Идеально работает для "summarize this podcast." Проваливается для:
- "What's the mood of this song?" — настроение в звуке, а не в словах.
- "Who is speaking, Alice or Bob?" — требуется speaker identification.
- "At what second does the explosion happen?" — temporal grounding теряется в тексте.
- "Is this real or generated audio?" — deepfake detection нужны acoustic features.

End-to-end сохраняет acoustic signal. Qwen-Audio и AF3 нативно обрабатывают музыку, environment и emotion.

### 2026 production recipe

Для нового audio-understanding product:

- Cascaded if: цель — transcription, нет music, нет emotion inference.
- AF3 / Qwen-Audio-family if: music, emotion, multi-speaker или complex audio reasoning.

Cascaded дешевле и проще. End-to-end более способен.

### MMAU — the audio reasoning benchmark

MMAU (Massive Multimodal Audio Understanding) — audio reasoning benchmark 2024-2025:

- 10,000 audio-text QA pairs по speech, music, environmental sounds.
- Охватывает classification, temporal reasoning, causal reasoning, open-ended QA.
- Проверяет то, что cascaded pipelines системно пропускают.

Open SOTA (AF3) на 0.72; proprietary frontier ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7). Разрыв меньше, чем open-vs-closed delta в VideoMME, что показывает созревание audio-LLMs.

## Использование

`code/main.py`:

- Реализует вычисление log-Mel spectrogram на stdlib: windowing, naive DFT, Mel filter-bank.
- Audio Q-former skeleton: по encoder output frames вычисляет Q, K, V, attention и выдает N tokens.
- Cascaded-vs-end-to-end comparison на toy task.

## Результат

Этот урок производит `outputs/skill-audio-llm-pipeline-picker.md`. По audio task (transcription, music tagging, emotion inference, multi-speaker diarization, environment classification) он выбирает cascaded, end-to-end AF3 или hybrid.

## Упражнения

1. Вычислите размерность log-Mel spectrogram для 30-second clip при 16kHz, 25ms window, 10ms hop, 80 Mel bins. Как это изменится при 48kHz?

2. Почему Whisper хуже работает на music? Какие audio features захватывает BEATs, которых нет у Whisper?

3. Audio Q-former with 64 queries vs 32: при какой сложности задач 64 окупается? Для чего 32 экономит compute?

4. Прочитайте AF3 Section 4 про on-demand thinking. Предложите три audio tasks, где chain-of-thought помогает сильнее всего.

5. Реализуйте минимальный diarization pipeline, используя output AF3. Как вы сигнализируете speaker changes?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "Mel features" | 2D (time, frequency) array значений log-magnitude после Mel filter banks |
| Audio Q-former | "Audio Perceiver" | Cross-attention bottleneck от output audio encoder к fixed-length queries, подаваемым в LLM |
| Cascaded | "ASR-then-LLM" | Pipeline, где Whisper транскрибирует, а text LLM рассуждает; теряет acoustic information |
| End-to-end | "Audio-LLM" | Audio features входят в LLM напрямую через Q-former; сохраняет acoustic signal |
| BEATs | "Audio AudioSet encoder" | SSL transformer, обученный на AudioSet; силен в music + environmental sounds |
| MMAU | "Audio reasoning bench" | 10k QA pairs по speech, music, environment; стандарт eval 2024 года |
| On-demand thinking | "Audio CoT" | Модель может опционально выдавать reasoning tokens перед final answer, повышая accuracy на 3-5 pts |

## Дополнительное чтение

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
