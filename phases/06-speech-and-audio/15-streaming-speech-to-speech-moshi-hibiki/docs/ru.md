# Streaming Speech-to-Speech — Moshi, Hibiki и full-duplex диалог

> 2024-2026 переопределили voice AI. Moshi поставляет одну модель, которая одновременно слушает и говорит с задержкой 200 ms. Hibiki делает speech-to-speech translation chunk-by-chunk. Обе отказываются от ASR → LLM → TTS pipeline в пользу единой full-duplex architecture поверх Mimi codec tokens. Это новый reference design.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 6 · 13 (Neural Audio Codecs), Фаза 6 · 11 (Real-Time Audio), Фаза 7 · 05 (Full Transformer)
**Время:** ~75 минут

## Цели обучения

- Объяснять, как Moshi слушает и говорит одновременно с латентностью около 200 мс.
- Описывать, почему поток внутреннего монолога текстом помогает full-duplex речи.
- Размещать Hibiki (потоковый перевод речь-в-речь) и Sesame CSM в стеке 2026.

## Проблема

Каждый voice agent из уроков 11 + 12 имеет фундаментальный latency floor около 300-500 ms: VAD срабатывает, STT обрабатывает, LLM рассуждает, TTS генерирует. У каждой стадии своя минимальная задержка. Можно настраивать и распараллеливать, но форма pipeline ограничивает вас.

Moshi (Kyutai, 2024-2026) спрашивает иначе: что, если pipeline нет? Что, если одна модель принимает аудио на вход и напрямую, непрерывно выдает аудио на выход, а text — только промежуточный "inner monologue", не обязательная стадия?

Ответ — **full-duplex speech-to-speech**. Теоретическая latency 160 ms (80 ms Mimi frame + 80 ms acoustic delay). Практическая latency 200 ms на одной L4 GPU. Это вдвое меньше, чем у лучшего pipelined voice agent.

## Концепция

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Архитектура Moshi

**Inputs.** Два Mimi codec streams, оба 12.5 Hz × 8 codebooks:

- Stream 1: пользовательское аудио (Mimi-encoded, постоянно поступает)
- Stream 2: собственное аудио Moshi (сгенерированное Moshi)

**Transformer.** Temporal Transformer на 7B parameters обрабатывает оба streams и text stream "inner monologue". На каждом шаге 80 ms он:

1. Потребляет последние пользовательские Mimi tokens (8 codebooks).
2. Потребляет самые свежие Moshi Mimi tokens (8 codebooks, по мере генерации).
3. Генерирует следующий text token Moshi (inner monologue).
4. Генерирует следующие Moshi Mimi tokens (8 codebooks через маленький Depth Transformer).

Все три streams — user audio, Moshi audio, Moshi text — идут параллельно. Moshi слышит пользователя, пока говорит; может прервать себя, когда пользователь перебивает; может делать back-channel ("mhm"), не ломая основную реплику.

**Depth transformer.** Внутри frame 8 codebooks не предсказываются параллельно — между ними есть зависимости. Маленький 2-layer "depth transformer" предсказывает их последовательно внутри 80 ms. Это стандартная факторизация для AR codec LMs (также VALL-E, VibeVoice).

### Почему inner-monologue text помогает

Без явного text модель должна неявно моделировать language в acoustic stream. Инсайт Moshi: заставить ее выдавать text tokens рядом с audio. Text stream фактически transcript того, что говорит Moshi. Это улучшает semantic coherence, упрощает замену language model head и дает transcripts бесплатно.

### Hibiki: streaming speech-to-speech translation

Та же архитектура, обученная на translation pairs. Исходное аудио поступает на вход, аудио на целевом языке непрерывно выходит наружу. Hibiki-Zero (Feb 2026) убирает необходимость word-level aligned training data — использует sentence-level data + GRPO reinforcement learning для оптимизации latency.

Сначала поддерживались четыре языковые пары; адаптация к новому языку требует ≈1000 часов.

### Более широкий стек Kyutai (2026)

- **Moshi** — full-duplex dialogue (сначала French, English хорошо поддержан)
- **Hibiki / Hibiki-Zero** — simultaneous speech translation
- **Kyutai STT** — streaming ASR (500 ms или 2.5 s look-ahead)
- **Kyutai Pocket TTS** — 100M-param TTS на CPU (Jan 2026)
- **Unmute** — полный pipeline, объединяющий эти компоненты на публичных серверах

Throughput на L40S GPU: 64 concurrent sessions на 3× real-time.

### Sesame CSM — родственник

Sesame CSM (2025) использует похожую идею — Llama-3 backbone с Mimi codec head. Но CSM однонаправленная (принимает context + text, выдает speech), а не full-duplex. Это лучший TTS с "voice presence" на рынке; не то же самое, что full-duplex capability Moshi.

### Числа производительности 2026

| Модель | Latency | Сценарий | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | то же | 5 language-pairs, без aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

## Соберите это

### Шаг 1: interface

Moshi exposes WebSocket server, который принимает 80 ms chunks Mimi-encoded audio и возвращает 80 ms chunks Mimi-encoded audio. В обе стороны. Постоянно.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Шаг 2: full-duplex loop

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Оба направления работают одновременно. Python asyncio или Rust futures — стандартный transport.

### Шаг 3: training objective (концептуально)

Для каждого 80 ms frame `t`:

- Input: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Predict: `moshi_text[t]`, затем `moshi_mimi[t, codebook_0..7]`

Text предсказывается до audio (inner monologue); audio — последовательно по codebooks внутри depth transformer.

### Шаг 4: где Moshi выигрывает и где нет

Moshi выигрывает:

- End-to-end меньше 250 ms на дешевом hardware.
- Естественные back-channels и interruptions.
- Нет glue code для pipeline.

Moshi не выигрывает:

- Tool calling (не обучен для этого; нужен отдельный LLM path).
- Длинное reasoning (Moshi — 8B-ish dialogue model, не Claude/GPT-4).
- Фактическая точность на нишевых темах.
- Большинство production enterprise use cases (в 2026 все еще используют pipelines).

## Используйте это

| Ситуация | Выбор |
|-----------|------|
| Voice companion с минимальной latency | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi, CSM |
| Enterprise agent with tools | Pipeline (урок 12), не Moshi |
| Custom-voice TTS in context | Sesame CSM |
| Speech-to-speech, любые языки | GPT-4o Realtime или Gemini 2.5 Live (commercial) |

## Ловушки

- **Ограниченный tool calling.** Moshi — dialogue model, не agent framework. Для tools объединяйте с pipeline.
- **Specific-voice conditioning.** Moshi использует одну обученную persona; cloning — отдельный training run.
- **Language coverage.** French + English excellent; others limited. Hibiki-Zero помогает, но все равно нужны training data.
- **Resource cost.** Full Moshi session занимает GPU slot; это не дешевый shared-tenant deploy pattern.

## Доведите до результата

Сохраните как `outputs/skill-duplex-pipeline.md`. Выберите pipeline или full-duplex architecture для workload voice agent, с обоснованием.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он символически симулирует архитектуру two-stream + inner-monologue.
2. **Средне.** Скачайте Moshi с HuggingFace, запустите server, протестируйте conversation. Измерьте wall-clock latency от конца речи пользователя до начала ответа Moshi.
3. **Сложно.** Возьмите pipeline agent из урока 12 и сравните P50 latency против Moshi на 20 matched test utterances. Опишите, когда pipeline все равно архитектурно выигрывает.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Full-duplex | Слушает и говорит одновременно | Два audio streams активны одновременно в одной модели. |
| Inner monologue | Text stream модели | Moshi выдает text tokens рядом с audio output. |
| Depth transformer | Inter-codebook predictor | Маленький transformer, предсказывающий 8 codebooks внутри одного 80 ms frame. |
| Mimi | Codec Kyutai | 12.5 Hz × 8 codebooks; semantic+acoustic; powers Moshi. |
| Streaming S2S | Audio → audio live | Chunk-by-chunk translation/dialogue без стадий pipeline. |
| Back-channeling | Реакции "mhm" | Moshi может выдавать короткие подтверждения, не ломая свою реплику. |

## Дополнительное чтение

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — статья.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — streaming translation без aligned data.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM spec.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — установка + server.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — закрытый коммерческий аналог.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — STT/TTS framework под капотом.
