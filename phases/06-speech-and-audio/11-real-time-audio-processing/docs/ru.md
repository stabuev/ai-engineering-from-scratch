# Обработка аудио в реальном времени

> Batch-пайплайны обрабатывают файл. Real-time пайплайны обрабатывают следующие 20 миллисекунд до прихода следующих 20. Каждый conversational AI, broadcast studio и telephony bot зависит от этого latency budget.

**Тип:** Сборка
**Языки:** Python, Rust
**Предварительные требования:** Фаза 6 · 02 (спектрограммы), Фаза 6 · 04 (ASR), Фаза 6 · 07 (TTS)
**Время:** ~75 минут

## Проблема

Вы хотите голосового ассистента, который ощущается живым. Человеческая задержка смены реплик — ~230 ms (silence-to-response). Все выше 500 ms звучит роботизированно; выше 1500 ms — сломанно. Budget полного цикла **hear → understand → respond → speak** в 2026:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) показал 200 ms full-duplex. GPT-4o-realtime (2024) — ~320 ms. Cascaded pipelines в 2022 выпускались с 2500 ms. Улучшение 10× пришло от трех техник: (1) streaming везде, (2) asynchronous pipelining с partial results, (3) interruptible generation.

## Концепция

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.** Real-time audio идет фиксированными блоками. Обычный выбор: 20 ms (320 samples при 16 kHz). Все ниже по пайплайну должно выдерживать этот cadence.

**Ring buffer.** Circular buffer фиксированного размера. Producer thread пишет новые frames, consumer thread читает. Предотвращает allocations в hot path. Размер ≈ maximum-latency × sample-rate; 2-секундный ring при 16 kHz = 32,000 samples.

**VAD (Voice Activity Detection).** Отключает downstream work, когда никто не говорит. Silero VAD 4.0 (2024) работает <1 ms на 30 ms frame на CPU. `webrtcvad` — более старая альтернатива.

**Streaming ASR.** Модели, выдающие partial transcripts по мере прихода аудио. Parakeet-CTC-0.6B в streaming mode (NeMo, 2024) дает 2–5% WER при 320 ms latency. Whisper-Streaming (Macháček et al., 2023) режет Whisper на chunks для near-streaming при ~2 s latency.

**Interruption.** Когда пользователь говорит, пока ассистент говорит, нужно (a) обнаружить barge-in, (b) остановить TTS, (c) отбросить оставшийся LLM output. Все за 100 ms, иначе пользователь воспринимает ассистента как «глухого».

**WebRTC Opus transport.** 20 ms frames, 48 kHz, adaptive bitrate 8–128 kbps. Стандарт для browser и mobile. LiveKit, Daily.co, Pion — стеки 2026 для voice apps.

**Jitter buffer.** Network packets приходят не по порядку / поздно. Jitter buffer переупорядочивает и сглаживает; слишком малый → audible gaps, слишком большой → latency. Типично 60–80 ms.

### Частые gotchas

- **Thread contention.** Python GIL + тяжелые модели могут starvation аудиопоток. Используйте C-callback audio library (sounddevice, PortAudio) и держите Python вне hot path.
- **Sample-rate conversion latency.** Resampling внутри pipeline добавляет 5–20 ms. Resample upfront или используйте zero-latency resampler (PolyPhase, `soxr_hq`).
- **TTS priming.** Даже быстрый TTS вроде Kokoro имеет 100–200 ms warm-up на первый запрос. Кэшируйте модель и прогрейте dummy run.
- **Echo cancellation.** Без AEC TTS output возвращается в mic и запускает ASR на голосе бота. WebRTC AEC3 — open-source default.

## Соберите это

### Шаг 1: ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity определяет максимальную buffering latency. 32,000 samples при 16 kHz = 2 s.

### Шаг 2: VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

В продакшене замените на Silero VAD:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Шаг 3: streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Шаг 4: interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Все держится на async I/O и cancellable TTS streaming. WebRTC peerconnection.stop() на audio track — canonical way.

## Используйте это

Стек 2026:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) или Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B или Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro или ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API или Moshi |

## Ловушки

- **Buffering 500 ms to be safe.** Buffer *is* your latency floor. Уменьшайте его.
- **Not pinning threads.** Audio callback на priority-lower-than-UI thread = glitches under load.
- **TTS chunks too small.** Sub-200 ms chunks делают vocoder artifacts слышимыми. 320 ms — sweet spot.
- **No jitter buffer.** Реальные сети jittery; без smoothing получите pops.
- **Single-shot error handling.** Audio pipelines должны быть crash-proof. Одно exception убивает session.

## Доведите до результата

Сохраните как `outputs/skill-realtime-designer.md`. Спроектируйте real-time audio pipeline с конкретными latency budgets по стадиям.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Симулирует ring buffer + energy VAD; печатает stage latencies для fake 10-second stream.
2. **Средне.** Используя `sounddevice`, соберите passthrough loop, который обрабатывает mic в 20 ms frames и печатает VAD state на каждом frame.
3. **Сложно.** Соберите full duplex echo test с `aiortc`: browser → WebRTC → Python → WebRTC → browser. Измерьте glass-to-glass latency 1 kHz pulse.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Ring buffer | Circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO для audio frames. |
| VAD | Silence gate | Модель или heuristic, маркирующая speech vs non-speech. |
| Streaming ASR | Real-time STT | Выдает partial text по мере прихода audio; bounded lookahead. |
| Jitter buffer | Network smoother | Queue для out-of-order packets; типично 60–80 ms. |
| AEC | Echo cancellation | Вычитает speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User и bot могут говорить одновременно; Moshi is full duplex. |

## Дополнительное чтение

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — production audio agent orchestration.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — echo cancellation в open source.
