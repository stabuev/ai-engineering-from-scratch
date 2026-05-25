# Сборка пайплайна голосового ассистента — капстоун Фазы 6

> Все из уроков 01-11, сшитое вместе. Соберите голосового ассистента, который слушает, рассуждает и отвечает голосом. В 2026 году это решенная инженерная задача, не исследовательская, но детали интеграции решают, выйдет ли она в продакшен.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 04, 05, 06, 07, 11; Фаза 11 · 09 (Function Calling); Фаза 14 · 01 (Agent Loop)
**Время:** ~120 минут

## Проблема

Соберите end-to-end ассистента:

1. Захватывает вход с микрофона (16 kHz mono).
2. Детектирует начало/конец речи пользователя.
3. Транскрибирует в streaming-режиме.
4. Передает transcript в LLM, которая может вызывать tools (timer, weather, calendar).
5. Стримит text LLM в TTS.
6. Воспроизводит audio пользователю.
7. Останавливается, если пользователь прерывает в середине ответа.

Целевая задержка: первый аудиобайт TTS в течение 800 ms после окончания utterance пользователя на laptop CPU. Целевое качество: без пропущенных слов, без hallucinated subtitles на тишине, без утечки voice cloning, без успешного prompt injection.

## Концепция

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### Семь компонентов

1. **Audio capture.** Mic → 16 kHz mono → 20 ms chunks. Обычно `sounddevice` в Python или native AudioUnit/ALSA/WASAPI в продакшене.
2. **VAD (Lesson 11).** Silero VAD @ threshold 0.5, min speech 250 ms, silence hang-over 500 ms. Сигналы "start" и "end."
3. **Streaming STT (Lesson 4-5).** Whisper-streaming, Parakeet-TDT или Deepgram Nova-3 (API). Partial + final transcripts.
4. **LLM with tool calling.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON schema для tools. Stream tokens.
5. **Streaming TTS (Lesson 7).** Kokoro-82M (самый быстрый open) или Cartesia Sonic (commercial). Стартуйте TTS после 20 LLM tokens.
6. **Playback.** Вывод на speaker; opus-encode для low-bandwidth networks.
7. **Interruption handler.** Если VAD сработал во время TTS playback, остановить playback, отменить LLM, перезапустить STT.

### Три failure modes, которые вы встретите

1. **First-word clip.** VAD стартует чуть поздно. Пользовательское "hey" пропадает. Start threshold 0.3, а не 0.5.
2. **Mid-response interrupt confusion.** LLM продолжает генерировать после interrupt; ассистент говорит поверх пользователя. Свяжите VAD → cancel-LLM.
3. **Silence hallucination.** Whisper выдает "Thanks for watching" на silent warm-up frames. Всегда используйте VAD-gate.

### Production reference stacks 2026

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | Industry default 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Одна модель; другая архитектура, урок 15 |
| Vapi / Retell (managed) | 300-500 ms | commercial | Fastest to launch; limited customization |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

## Соберите это

### Шаг 1: mic capture с chunking (pseudocode)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Шаг 2: VAD-gated turn capture

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Шаг 3: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Шаг 4: tool calling внутри LLM loop

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Шаг 5: interruption handling

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Используйте это

См. `code/main.py`: там runnable simulation, которая соединяет все семь компонентов со stub models, чтобы показать форму pipeline даже без hardware. Для реальной реализации замените stubs на:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` или `openai-whisper`
- `openai` (`gpt-4o`) или `anthropic`
- `kokoro` или `cartesia`
- `sounddevice` для I/O

## Ловушки

- **Logging PII forever.** Full-turn audio — PII в большинстве jurisdictions. 30-day retention, encrypted at rest.
- **No barge-in.** Пользователи будут прерывать. Ассистент должен остановиться.
- **TTS that blocks.** Synchronous TTS блокирует event loop. Используйте async или separate thread.
- **No tool-call error handling.** Tools fail. LLM должна получить error + один retry, затем gracefully degrade.
- **Overzealous hallucination filters.** Over-filter — ассистент повторяет "I can't help with that." Under-filter — говорит что угодно. Калибруйте на held-out set.
- **No wake-word option.** Always-listening — privacy liability. Добавьте wake-word gate (Porcupine или openWakeWord).

## Доведите до результата

Сохраните как `outputs/skill-voice-assistant-architect.md`. По budget + scale + language + compliance constraints выдайте полную спецификацию stack.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он симулирует один full turn end-to-end со stub modules и печатает latency по стадиям.
2. **Средне.** Замените STT stub на реальную Whisper model на заранее записанном `.wav`. Измерьте WER и end-to-end latency.
3. **Сложно.** Добавьте tool calling: реализуйте `get_weather` (любой API) и `set_timer`. Пропустите LLM через tools и проверьте, что при "set a 5 minute timer" вызывается правильная функция и spoken reply подтверждает это.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Turn | User + assistant round-trip | Один VAD-bounded user speech + один LLM-TTS response. |
| Barge-in | Interruption | Пользователь говорит, пока ассистент говорит; ассистент останавливается. |
| Wake word | "Hey assistant" | Короткий keyword detector; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Turn ending | VAD + min-silence decision, что пользователь закончил. |
| Pre-roll | Pre-speech buffer | Держать 200-400 ms audio до VAD, чтобы не обрезать first word. |
| Tool call | Function invocation | LLM выдает JSON; runtime dispatches; result feeds back in-loop. |

## Дополнительное чтение

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — production-grade reference.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — managed voice-native path.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference (Lesson 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling.
