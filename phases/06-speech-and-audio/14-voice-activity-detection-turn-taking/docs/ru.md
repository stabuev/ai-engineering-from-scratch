# Детекция речевой активности (VAD) и смена реплик — Silero, Cobra и flush trick

> Каждый voice agent живет или умирает на двух решениях: говорит ли пользователь сейчас и закончил ли он? VAD отвечает на первое. Turn-detection (VAD + silence-hangover + semantic endpoint model) отвечает на второе. Ошибитесь, и ассистент либо перебивает пользователей, либо никогда не замолкает.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 11 (Real-Time Audio), Фаза 6 · 12 (Voice Assistant)
**Время:** ~45 минут

## Цели обучения

- Реализовывать трехуровневый VAD-каскад от энергетического гейта до Silero VAD.
- Настраивать определение конца реплики через silence hangover и семантический эндпоинт.
- Объяснять flush-трюк и читать метрики VAD (TPR при FPR 5%).

## Проблема

Три разных решения, которые voice agent принимает на каждом 20 ms chunk:

1. **Есть ли речь в этом фрейме?** — VAD. Бинарно, по каждому фрейму.
2. **Начал ли пользователь новое высказывание?** — onset detection.
3. **Закончил ли пользователь?** — end-pointing (turn-end).

Наивный ответ (energy threshold) ломается на любом шуме — traffic, keyboards, crowd babble. Ответ 2026 года: Silero VAD (open, deep-learned) + turn-detection model (semantic endpointing) + VAD-calibrated silence hangover.

## Концепция

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### Трехуровневый VAD cascade

**Tier 1: energy gate.** Самый дешевый. Threshold RMS на -40 dBFS. Отсекает очевидную тишину, но срабатывает на любой шум выше threshold.

**Tier 2: Silero VAD** (2020-2026, MIT). 1M parameters. Обучен на 6000+ languages. Работает ~1 ms на 30 ms chunk на одном CPU thread. 87.7% TPR при 5% FPR. Open-source default.

**Tier 3: semantic turn detector.** LiveKit turn-detection model (2024-2026) или свой small classifier. Отличает паузу в середине предложения от завершенной реплики. Использует linguistic context (интонация + последние слова), не только тишину.

### Ключевые параметры и defaults

- **Threshold.** Silero выдает probability; классифицируйте речь при &gt; 0.5 (default) или &gt; 0.3 (sensitive). Ниже threshold = меньше обрезаний первого слова, больше false positives.
- **Minimum speech duration.** Отбрасывайте речь короче 250 ms — обычно это кашель или шум стула.
- **Silence hangover (end-pointing).** После VAD=0 ждите 500-800 ms, прежде чем объявить end-of-turn. Слишком коротко → перебьете пользователя. Слишком долго → sluggish.
- **Pre-roll buffer.** Держите 300-500 ms audio до срабатывания VAD. Не дает обрезать "hey".

### Flush trick (Kyutai 2025)

Streaming STT models имеют look-ahead delay (500 ms для Kyutai STT-1B, 2.5 s для STT-2.6B). Обычно после end-of-speech нужно ждать столько же для transcript. Flush trick: когда VAD дает end-of-speech, **отправьте flush signal в STT**, заставив немедленный output. STT работает примерно на 4× realtime, поэтому 500 ms buffer завершается за ~125 ms.

End-to-end: 125 ms VAD + flush STT = разговорная задержка.

### Сравнение VAD 2026

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero — правильный default. Cobra — compliance / accuracy upgrade. Energy-only VAD не место в production 2026.

## Соберите это

### Шаг 1: energy gate

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Шаг 2: Silero VAD in Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Шаг 3: turn-end state machine

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Шаг 4: skeleton flush trick

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) должен поддерживать flush. Whisper streaming не поддерживает — он block-based и всегда ждет chunks.

## Используйте это

| Ситуация | Выбор VAD |
|-----------|-----------|
| Open, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device (phone) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |
| Нужное качество turn-ending | Silero + LiveKit turn-detector layered |

Правило: никогда не ship energy-only VAD, если есть хоть какая-то альтернатива.

## Ловушки

- **Fixed threshold.** Работает в тишине, ломается в шуме. Калибруйте on-device или переходите на Silero.
- **Too-short silence hangover.** Agent перебивает в середине фразы. 500-800 ms — sweet spot для conversational speech.
- **Too-long hangover.** Ощущается sluggish. Проводите A/B test с целевыми пользователями.
- **No pre-roll buffer.** Первые 200-300 ms user audio теряются. Всегда держите rolling pre-roll.
- **Ignoring semantic endpointing.** "Hmm, let me think..." содержит длинные pauses. Пользователи не любят, когда их обрывают на середине мысли. Используйте LiveKit turn-detector или аналог.

## Доведите до результата

Сохраните как `outputs/skill-vad-tuner.md`. Выберите VAD model, threshold, hangover, pre-roll и turn-detection strategy для workload.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он симулирует sequence speech + silence + speech + coughs и тестирует три VAD tiers.
2. **Средне.** Установите `silero-vad`, обработайте 5-минутную запись, настройте threshold, минимизируя first-word clips и false triggers. Сообщите precision/recall.
3. **Сложно.** Соберите mini turn-detector: Silero VAD + 3-layer MLP на embeddings последних 10 words (используйте sentence-transformers). Обучите на hand-labeled turn-end dataset. Побейте Silero-only на 10% F1.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| VAD | Voice detector | Бинарная per-frame оценка: есть ли здесь речь? |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint. |
| Silence hangover | Wait-after-speech | Время ожидания перед turn end; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Держать 300-500 ms audio до срабатывания VAD. |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms вместо 500 ms delay. |
| Semantic endpoint | "Они правда закончили?" | ML classifier, смотрящий на words, а не только silence. |
| TPR @ FPR 5% | ROC point | Стандартный VAD benchmark; 87.7% для Silero, 50% WebRTC. |

## Дополнительное чтение

- [Silero VAD](https://github.com/snakers4/silero-vad) — референсный open VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — коммерческий лидер по accuracy.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — sub-200 ms engineering trick.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — semantic endpointing в production.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — legacy baseline.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — diarization-grade segmentation.
