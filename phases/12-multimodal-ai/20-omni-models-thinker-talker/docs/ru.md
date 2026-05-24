# Omni-модели: Qwen2.5-Omni и разделение Thinker-Talker

> Продуктовая демонстрация GPT-4o в мае 2024 года была прорывной не из-за базовой модели, а из-за формы продукта — голосового интерфейса, где вы говорите, модель видит то, что видит камера, и отвечает голосом менее чем за 250ms. Открытая экосистема провела остаток 2024 и 2025 годов в гонке за этой продуктовой поверхностью. Qwen2.5-Omni (март 2025) — референсный открытый дизайн: Thinker (большой text-generating transformer) плюс Talker (параллельный speech-generating transformer), связанные потоковыми речевыми токенами. Mini-Omni упростила его, Moshi сравнялась по задержке, GLM-4-Voice расширила его на китайский язык. В этом уроке разбирается архитектура Thinker-Talker и latency budget, благодаря которому потоковый диалог реального времени работает.

**Тип:** Практика
**Языки:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Предварительные требования:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Время:** ~180 минут

## Цели обучения

- Разделить inference pipeline на Thinker (текстовое рассуждение) и Talker (синтез речи) и объяснить, почему параллельный стриминг работает.
- Вычислить time-to-first-audio-byte (TTFAB) budget для разговорного взаимодействия, компонент за компонентом.
- Описать time-aligned position encoding TMRoPE по vision, audio и text внутри Thinker.
- Назвать три real-time conversational patterns: half-duplex, turn-taking, full-duplex.

## Проблема

Голосовой ассистент реального времени должен делать многое и быстро:

1. Слышать пользователя. Real-time speech tokenization, voice activity detection (VAD), чтобы понять, когда он закончил говорить.
2. Опционально видеть. Camera input при 2-4 FPS, streamed в Thinker вместе с audio.
3. Думать. Составить response, conditioned on conversation history.
4. Говорить. Синтезировать audio tokens, decode to waveform, stream в динамики пользователя.

Каждый шаг добавляет задержку. Для ощущения разговора нужен total round-trip < 500ms — ниже этого пользователь перестает замечать lag. GPT-4o заявляет ~250ms. Moshi ~160ms. Qwen2.5-Omni ~350-500ms.

Каждый компонент должен стримить. Нельзя "batch everything then decode."

## Концепция

### Thinker and Talker

Декомпозиция Qwen2.5-Omni:

- Thinker: text-generating transformer на 7B-80B. Потребляет interleaved text + image + audio tokens. Выдает text tokens, представляющие, что сказать.
- Talker: меньший speech-generating transformer (200M-1B). Потребляет text output tokens Thinker плюс recent speech-context tokens. Выдает discrete speech tokens (residual-VQ indices).
- Speech decoder: streaming waveform decoder (семейство SNAC, MoVQGAN), который в реальном времени преобразует speech tokens в audio samples.

Разделение важно. Thinker должен быть большим для хорошего reasoning. Talker может быть маленьким, потому что его задача локальна — convert text to speech tokens. Более крупный Talker не выразительнее; он медленнее.

Параллельный запуск:

1. Thinker выдает text token t_i.
2. Talker потребляет t_i (via streaming) и выдает speech tokens s_i, s_{i+1}, ..., s_{i+k}.
3. Speech decoder потребляет speech tokens по мере поступления и выдает audio samples.
4. К тому времени, когда Thinker находится на text token t_{i+3}, Talker уже застримил audio для t_0..t_{i+2}.

### TMRoPE — time-aligned multimodal positions

Thinker должен интегрировать image frames (поступающие, скажем, при 4 FPS), audio frames (поступающие с 50 frames/second) и text из conversation history. Наивный sequence order (сначала все images, затем все audio, затем text) теряет temporal alignment.

TMRoPE присваивает абсолютные timestamps каждому token. Vision token at t=2.3s. Audio token at t=2.32s. Text token from the user "stop" at t=2.35s. RoPE вращает attention по timestamp; модель видит их как temporally concurrent.

Это инфраструктура, благодаря которой работает "he waved while saying hello" — модель видит video frame и audio в один концептуальный момент.

### Streaming speech synthesis

Speech tokens должны стримиться. Mini-Omni (Xie & Wu, 2024) ввела "language models can hear, talk while thinking in streaming": Thinker output tokens и Talker output tokens чередуются в одной sequence. Talker срабатывает сразу, как только Thinker коммитит следующий text token. Без batch boundaries.

Moshi (Défossez et al., октябрь 2024) — самая быстрая открытая реализация. 160ms TTFAB на одном A100. Архитектура: один 7B transformer, который выдает text и speech tokens на чередующихся позициях, с "inner monologue", разделяющим thinking stream и speaking stream. Это фактически Thinker + Talker, слитые в одну модель с аккуратным обучением.

### VAD and turn-taking

Voice activity detection работает на входной стороне. Два паттерна:

- Half-duplex: пользователь говорит, модель слушает. Модель говорит, пользователь слушает. Четкая передача хода через VAD silence detection (~200ms).
- Full-duplex: оба могут говорить одновременно. Модель может backchannel ("uh-huh") или interrupt. Намного сложнее. Moshi поддерживает это.

Qwen2.5-Omni по умолчанию поддерживает half-duplex, с turn-taking через silence threshold. Full-duplex требует application-layer handling.

### Qwen3-Omni (November 2025)

Преемник. Qwen3-80B Thinker, более крупный Talker, улучшенный TMRoPE-v2. Latency близка к 250ms у GPT-4o. Open weights. Benchmarks на OmniBench конкурентны с Gemini 2.0 Live.

### Production latency budget

Для типичного streaming interaction:

- Mic -> audio tokens: 40-80ms.
- Prefill (prompt + history): 100-200ms на 7B, гораздо больше на 70B.
- First Thinker text token: 40ms.
- Talker processes first text token: 20ms.
- First speech tokens commit: 40ms.
- Residual-VQ decode: 30ms.
- Speech waveform decode: 50-80ms.

Total TTFAB: 320-510ms на 7B, 600-900ms на 70B. Frontier quality обычно означает 70B+; отсюда frontier latency gap.

### Token-rate math

Для речи 16kHz с base speech tokens 50 Hz нужно 50 speech tokens per second выхода. Talker должен выдавать ≥50 tok/s, чтобы успевать. При типичном throughput LLM 30-80 tok/s на H100 маленький (200-300M) Talker достаточно быстр; 7B Talker начал бы отставать.

Именно поэтому существуют маленькие dedicated Talker models, а не "just use the main model."

## Использование

`code/main.py`:

- Симулирует Thinker-Talker pipeline с mock token-emission rates.
- Вычисляет TTFAB для настраиваемых model sizes и mic sample rates.
- Демонстрирует half-duplex turn-taking с VAD silence threshold.

## Результат

Этот урок производит `outputs/skill-omni-streaming-budget.md`. По target TTFAB и feature set (vision-in, bilingual, full-duplex) продукта real-time voice он выбирает Qwen2.5-Omni, Qwen3-Omni, Moshi или Mini-Omni и задает размеры Thinker/Talker.

## Упражнения

1. Ваш target TTFAB — 300ms. На 7B Thinker и 300M Talker распишите задержку каждого компонента.

2. Qwen2.5-Omni использует TMRoPE. Опишите, что видит модель для prompt, где пользователь начинает говорить при t=1s, а камера фиксирует gesture при t=1.2s.

3. Поддержка full-duplex требует, чтобы модель выдавала audio while listening. Предложите training data format, который этому учит.

4. Прочитайте Section 4 статьи Moshi. Опишите разделение "inner monologue" и почему оно избегает Thinker-Talker split.

5. Вычислите throughput budget: как быстро Talker должен выдавать tokens, чтобы успевать за речью 16kHz при 50 base-layer tokens/sec?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Thinker | "Reasoning brain" | Большой text-generating transformer, производящий то, что сказать |
| Talker | "Speech-generating mouth" | Маленький transformer, производящий discrete speech tokens из текста Thinker |
| TTFAB | "Latency budget" | Time-to-first-audio-byte: от конца речи пользователя до первого audio sample out |
| TMRoPE | "Time-aligned RoPE" | Position encoding с absolute timestamps по vision, audio, text |
| Half-duplex | "Turn-taking" | Пользователь и модель чередуются; VAD silence определяет user-done |
| Full-duplex | "Simultaneous" | Модель может говорить и слушать одновременно; способна к backchannel |
| Inner monologue | "Moshi separation" | Single-model design, где thinking-stream и speaking-stream чередуются |

## Дополнительное чтение

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
