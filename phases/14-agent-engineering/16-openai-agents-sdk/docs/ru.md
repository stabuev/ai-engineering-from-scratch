# OpenAI Agents SDK: передачи, guardrails и трассировка

> OpenAI Agents SDK — легковесный мультиагентный фреймворк, построенный на Responses API. Пять примитивов: Agent, Handoff, Guardrail, Session, Tracing. Handoffs — это tools с именами `transfer_to_<agent>`. Guardrails срабатывают на input или output. Tracing включен по умолчанию.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 06 (Tool Use)
**Время:** ~75 минут

## Цели обучения

- Назвать пять примитивов OpenAI Agents SDK.
- Объяснить handoffs: почему они моделируются как tools, какую форму имени видит модель и как передается context.
- Отличать input guardrails, output guardrails и tool guardrails; объяснить `run_in_parallel` vs blocking mode.
- Реализовать stdlib runtime с handoffs + guardrails + tracing в стиле spans.

## Проблема

Agents, которые не умеют аккуратно делегировать, в итоге запихивают все в один prompt. Agents без guardrails отправляют PII, нарушающий policy output или зацикливаются навсегда. SDK OpenAI кодифицирует три примитива, которые делают multi-agent управляемым.

## Концепция

### Пять примитивов

1. **Agent.** LLM + instructions + tools + handoffs.
2. **Handoff.** Делегирование другому agent. Представляется модели как tool с именем `transfer_to_<agent_name>`.
3. **Guardrail.** Валидация input (только первый agent), output (только последний agent) или tool invocation (для каждого function tool).
4. **Session.** Автоматическая history диалога между turns.
5. **Tracing.** Встроенные spans для LLM generations, tool calls, handoffs, guardrails.

### Передачи как tools

Модель видит `transfer_to_billing_agent` в своем списке tools. Его вызов сообщает runtime:

1. Скопировать conversation context (или свернуть его через beta `nest_handoff_history`).
2. Инициализировать целевой agent с его instructions.
3. Продолжить run с целевым agent.

Это productized-версия supervisor pattern (Урок 13 / Урок 28).

### Guardrails

Три вида:

- **Input guardrails.** Запускаются на input первого agent. Отклоняют небезопасные или out-of-scope requests до любого LLM call.
- **Output guardrails.** Запускаются на output последнего agent. Ловят утечки PII, нарушения policy, malformed responses.
- **Tool guardrails.** Запускаются для каждого function-tool. Валидируют arguments, проверяют permissions, аудитируют execution.

Режим:

- **Parallel** (по умолчанию). Guardrail LLM работает параллельно с main LLM. Ниже tail latency. Если сработал tripwire, работа main LLM отбрасывается (потеря tokens).
- **Blocking** (`run_in_parallel=False`). Guardrail LLM запускается первым. Если сработал tripwire, tokens на main call не тратятся.

Tripwires выбрасывают `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Трассировка

Включен по умолчанию. Каждая LLM generation, tool call, handoff и guardrail испускает span. `OPENAI_AGENTS_DISABLE_TRACING=1` отключает это. `add_trace_processor(processor)` отправляет spans в ваш backend параллельно с OpenAI.

### Sessions

`Session` хранит conversation history в backend (SQLite, Redis, custom). `Runner.run(agent, input, session=session)` автоматически загружает и добавляет историю.

### Где этот паттерн ломается

- **Handoff drift.** Agent A передает Agent B, который передает обратно Agent A. Добавьте hop counter.
- **Guardrail bypass.** Tool guardrails срабатывают только на function tools; built-in tools (file reader, web fetch) требуют отдельной policy.
- **Over-tracing.** Sensitive content в spans. Сочетайте с правилами OTel GenAI content-capture (Урок 23): храните снаружи, ссылайтесь по ID.

## Соберите это

`code/main.py` реализует форму SDK на stdlib:

- `Agent`, `FunctionTool`, `Handoff` (как function tool с семантикой transfer).
- `Runner` с input/output/tool guardrails, handoff dispatch и hop counter.
- Простой span emitter, показывающий форму trace.
- Triage agent, который передает в billing или support на основе query пользователя; guardrail срабатывает на одном input.

Запустите:

```
python3 code/main.py
```

Trace показывает два успешных handoffs, один input guardrail trip и дерево spans, зеркалящее то, что испускает настоящий SDK.

## Используйте это

- **OpenAI Agents SDK** для OpenAI-first продуктов.
- **Claude Agent SDK** (Урок 17) для Claude-first продуктов.
- **LangGraph** (Урок 13), когда нужны явное state и durable resume.
- **Custom**, когда нужен точный контроль (voice, multi-provider, federated deployments).

## Отправьте в работу

`outputs/skill-agents-sdk-scaffold.md` создает scaffold приложения Agents SDK с triage agent, handoffs, input/output/tool guardrails, session store и trace processor.

## Упражнения

1. Добавьте handoff hop counter: отказывать после N transfers. Протрассируйте поведение.
2. Реализуйте `nest_handoff_history` как option — сворачивайте предыдущие messages в одну summary перед transfer.
3. Напишите blocking output guardrail. Сравните latency на prompts, которые его trip, и на тех, которые проходят.
4. Подключите `add_trace_processor` к JSON logger. Какую форму он испускает для каждого span?
5. Прочитайте документацию SDK. Перенесите свой stdlib toy в `openai-agents-python`. Что вы смоделировали неправильно?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Agent | "LLM + instructions" | Тип Agent в SDK; владеет tools и handoffs |
| Handoff | "Transfer" | Tool, который модель вызывает для делегирования другому agent |
| Guardrail | "Policy check" | Валидация input / output / tool invocation |
| Tripwire | "Guardrail trip" | Exception, выбрасываемый при отклонении guardrail |
| Session | "History store" | Conversation memory, сохраняемая между runs |
| Tracing | "Spans" | Встроенная observability над LLM + tool + handoff + guardrail |
| Blocking guardrail | "Sequential check" | Guardrail запускается первым; нет потери tokens при trip |
| Parallel guardrail | "Concurrent check" | Guardrail работает параллельно; ниже latency, tokens тратятся при trip |

## Дополнительное чтение

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — примитивы, handoffs, guardrails, tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — аналог в стиле Claude
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — когда вообще стоит использовать handoffs
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — стандарт, на который мапятся spans Agents SDK
