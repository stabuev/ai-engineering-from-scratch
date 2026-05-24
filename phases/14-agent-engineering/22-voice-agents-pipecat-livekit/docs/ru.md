# Голосовые агенты: Pipecat и LiveKit

> Голосовые агенты в 2026 году стали полноценной производственной категорией. Pipecat дает Python-конвейер на основе фреймов (VAD -> STT -> LLM -> TTS -> transport). LiveKit Agents связывает AI-модели с пользователями через WebRTC. Для премиальных стеков производственные цели по задержке находятся на уровне 450-600 ms end-to-end.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Время:** ~60 минут

## Цели обучения

- Описать фреймовый конвейер Pipecat: DOWNSTREAM (source→sink) и UPSTREAM (control).
- Назвать канонические этапы голосового конвейера и транспорты, которые поддерживает Pipecat.
- Объяснить два класса голосовых агентов LiveKit Agents (MultimodalAgent, VoicePipelineAgent) и когда подходит каждый из них.
- Суммировать производственные ожидания по задержке в 2026 году и то, как они определяют архитектурные решения.

## Проблема

Голосовые агенты - это не текстовый цикл с прикрученным TTS. Бюджеты задержки жесткие (~600 ms), частичный аудиопоток является нормой, определение хода - это модель, а транспорты варьируются от telephony SIP до WebRTC. Либо вы строите фреймовый конвейер (Pipecat), либо опираетесь на платформу (LiveKit).

## Концепция

### Pipecat (pipecat-ai/pipecat)

- Python-фреймворк конвейеров на основе фреймов.
- Цепочка `Frame` -> `FrameProcessor`.
- Два направления потока:
  - **DOWNSTREAM** — source -> sink (audio in, TTS out).
  - **UPSTREAM** — обратная связь и управление (cancellation, metrics, barge-in).
- `PipelineTask` управляет жизненным циклом через события (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) и observers для metrics/tracing/RTVI.

Типичный конвейер:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Транспорты: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows добавляет структурированные диалоги (state machines). Pipecat Cloud - это managed runtime.

### LiveKit Agents (livekit/agents)

- Связывает AI-модели с пользователями через WebRTC.
- Ключевые понятия: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Два класса голосовых агентов:
  - **MultimodalAgent** — прямое аудио через OpenAI Realtime или аналог.
  - **VoicePipelineAgent** — каскад STT -> LLM -> TTS; дает контроль на уровне текста.
- Семантическое определение хода через transformer model.
- Нативная интеграция MCP.
- Телефония через SIP.
- 50+ моделей без API keys через LiveKit Inference; еще 200+ через plugins.

### Коммерческие платформы

Vapi (~450-600 ms на оптимизированном premium stack) и Retell (~600 ms end-to-end по 180 тестовым звонкам) строятся поверх этих подходов. Выбирайте платформу, когда нужен managed voice stack без команды WebRTC.

### Где этот паттерн ломается

- **Нет обработки barge-in.** Пользователь перебивает; агент продолжает говорить. Нужны UPSTREAM cancel frames в Pipecat или эквивалент в LiveKit.
- **Игнорируется STT confidence.** Низкоуверенные транскрипты передаются в LLM как истина. Ставьте gate по confidence или просите подтверждение.
- **TTS обрывается посреди предложения.** Когда конвейер отменяется в середине высказывания, TTS должен узнать об этом или обрезать аудио.
- **Игнорируется бюджет задержки.** Каждый компонент добавляет 50-200 ms. Сложите всю цепочку до запуска в production.

### Типичные задержки 2026 года

- VAD: 20-60 ms
- STT partial: 100-250 ms
- LLM first token: 150-400 ms
- TTS first audio: 100-200 ms
- Transport RTT: 30-80 ms

End-to-end 450-600 ms - это premium. 800-1200 ms встречается часто. Все, что > 1500 ms, ощущается сломанным.

## Соберите это

`code/main.py` - игрушечный фреймовый конвейер с:

- Типами `Frame` (audio, transcript, text, tts_audio, control).
- Интерфейсом `Processor` с `process(frame)`.
- Пятиэтапным конвейером (VAD -> STT -> LLM -> TTS -> transport) как scripted processors.
- UPSTREAM cancel frame для демонстрации barge-in.

Запустите:

```
python3 code/main.py
```

Trace показывает нормальный поток и barge-in cancel, который останавливает TTS в середине высказывания.

## Используйте это

- **Pipecat** для полного контроля — custom processors, Python-first, pluggable providers.
- **LiveKit Agents** для WebRTC-first развертываний и телефонии.
- **Vapi / Retell** для hosted voice agents без команды WebRTC.
- **OpenAI Realtime / Gemini Live** для прямого audio-in/audio-out (MultimodalAgent).

## Доведите до продакшена

`outputs/skill-voice-pipeline.md` формирует каркас голосового конвейера в стиле Pipecat с VAD + STT + LLM + TTS + transport, а также обработкой barge-in.

## Упражнения

1. Добавьте metrics observer в игрушечный конвейер: считайте frames per stage per second. Где накапливается задержка?
2. Реализуйте STT с gate по confidence: ниже порога просите "could you repeat that?"
3. Добавьте семантическое определение хода: простое правило — если transcript заканчивается на "?", ход завершен.
4. Прочитайте transport docs Pipecat. Замените stdlib transport на конфиг SmallWebRTCTransport (stub).
5. Измерьте OpenAI Realtime и каскад STT+LLM+TTS на одном и том же запросе. Какую цену по задержке несет контроль на уровне текста?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Frame | "Событие" | Типизированная единица данных в конвейере (audio, transcript, text, control) |
| Processor | "Этап конвейера" | Обработчик с process(frame) |
| DOWNSTREAM | "Прямой поток" | От source к sink: аудио на входе, речь на выходе |
| UPSTREAM | "Поток обратной связи" | Управление: отмена, метрики, barge-in |
| VAD | "Определение голосовой активности" | Определяет, когда пользователь говорит |
| Semantic turn detection | "Умное завершение реплики" | Решение на основе модели о том, что пользователь закончил |
| MultimodalAgent | "Прямой аудиоагент" | Аудио на входе, аудио на выходе; без текста посередине |
| VoicePipelineAgent | "Каскадный агент" | STT + LLM + TTS; контроль на уровне текста |

## Дополнительное чтение

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — фреймовый конвейер, процессоры, транспорты
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + голосовые примитивы
- [Vapi](https://vapi.ai/) — управляемая голосовая платформа
- [Retell AI](https://www.retellai.com/) — управляемый голосовой стек с бенчмарками задержки
