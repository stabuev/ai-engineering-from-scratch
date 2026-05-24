# Agno и Mastra: production-runtimes

> Agno (Python) и Mastra (TypeScript) — пара production-runtime 2026 года. Agno нацелен на микросекундную инициализацию agent и stateless FastAPI backends. Mastra поставляет agents, tools, workflows, unified model routing и composite storage на основе Vercel AI SDK.

**Тип:** Изучение
**Языки:** Python, TypeScript
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 13 (LangGraph)
**Время:** ~45 минут

## Цели обучения

- Определить performance targets Agno и когда они важны.
- Назвать три примитива Mastra — Agents, Tools, Workflows — и поддерживаемые server adapters.
- Объяснить, почему stateless session-scoped FastAPI backend является рекомендуемым production-путем Agno.
- Выбрать Agno vs Mastra для заданного stack (Python-first vs TypeScript-first).

## Проблема

LangGraph, AutoGen, CrewAI тяжеловесны как frameworks. Команды, которым нужно "просто agent loop, быстро, в моем runtime", выбирают Agno (Python) или Mastra (TypeScript). Оба отдают часть framework-owned примитивов ради чистой скорости и более плотной посадки в окружающий stack.

## Концепция

### Agno

- Python runtime, ранее Phi-data.
- "No graphs, chains, or convoluted patterns — just pure python."
- Performance targets из их документации: ~2μs agent instantiation, ~3.75 KiB memory на agent, ~23 model providers.
- Production path: stateless session-scoped FastAPI backend. Каждый request запускает fresh agent; session state живет в DB.
- Native multimodal (text, image, audio, video, file) и agentic RAG.

Цели скорости важны, когда у вас тысячи short-lived agents в секунду (chat fan-in, evaluation pipelines). Они менее важны, когда один agent работает 10 минут.

### Mastra

- TypeScript, построен на Vercel AI SDK.
- Три примитива: **Agents**, **Tools** (Zod-typed), **Workflows**.
- Unified Model Router — 3,300+ models у 94 providers (март 2026).
- Composite storage: memory, workflows, observability в разные backends; ClickHouse рекомендуется для observability at scale.
- Apache 2.0 с директориями `ee/` под source-available enterprise license.
- Server adapters для Express, Hono, Fastify, Koa; first-class интеграция с Next.js и Astro.
- Поставляет Mastra Studio (localhost:4111) для отладки.
- 22k+ GitHub stars, 300k+ weekly npm downloads на версии 1.0 (январь 2026).

### Позиционирование

Ни один из них не пытается быть LangGraph. Они конкурируют по:

- **Language fit.** Agno для Python-first команд; Mastra для TypeScript-first.
- **Runtime ergonomics.** Agno = почти нулевой overhead; Mastra = интеграция с экосистемой Vercel.
- **Observability.** Оба интегрируются с Langfuse/Phoenix/Opik (Урок 24), но Mastra Studio является first-party.

### Когда что выбирать

- **Agno** — Python backend, много short-lived agents, жесткие perf requirements, FastAPI shop.
- **Mastra** — TypeScript backend, Next.js / Vercel deploy, unified multi-provider model routing, Zod-typed tools.
- **LangGraph** (Урок 13) — когда durable state и явное graph reasoning важнее raw speed.
- **OpenAI / Claude Agent SDK** — когда нужна productized shape провайдера (Уроки 16–17).

### Где этот паттерн ломается

- **Perf-for-perf's-sake.** Выбор Agno потому, что "2μs" звучит хорошо, хотя workload — один медленный agent call на request. Overhead не bottleneck.
- **Ecosystem lock-in.** Vercel-flavored интеграция Mastra — плюс на Vercel и минус в других местах.
- **Enterprise license confusion.** Директории `ee/` в Mastra source-available, а не Apache 2.0. Читайте licenses, если планируете fork.

## Соберите это

Этот урок в первую очередь сравнительный: один code artifact не сможет честно раскрыть оба frameworks. См. `code/main.py` для side-by-side toy: минимальный flow "run an agent, stream the output, persist session", реализованный дважды (один раз в форме Agno, один раз в форме Mastra).

Запустите:

```
python3 code/main.py
```

Две структурно разные, но функционально эквивалентные traces.

## Используйте это

- **Agno** — Python backend, которому нужна скорость и форма FastAPI.
- **Mastra** — TypeScript backend с множеством providers и workflow primitives.
- Оба поставляют first-party observability hooks. Оба интегрируются с Langfuse.

## Отправьте в работу

`outputs/skill-runtime-picker.md` выбирает Agno, Mastra, LangGraph или provider SDK на основе stack, latency budget и operational shape.

## Упражнения

1. Прочитайте документацию Agno. Перенесите stdlib ReAct loop (Урок 01) в Agno. Что исчезло? Что осталось?
2. Прочитайте документацию Mastra. Перенесите тот же loop в Mastra. Что изменилось в tool typing (Zod vs nothing)?
3. Benchmark: измерьте agent instantiation latency на своем stack. Важны ли 2μs Agno для вашего workload?
4. Спроектируйте migration: если вы запускали CrewAI в Python, что сломается при переходе на Agno?
5. Прочитайте license terms для `ee/` в Mastra. Какие restrictions повлияют на open-source fork?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Agno | "Быстрые Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents на Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Один client для 3,300+ models у 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability каждый в отдельный store |
| Mastra Studio | "Local debugger" | localhost:4111 UI для introspecting agents |
| Source-available | "Not OSS" | License разрешает чтение source, но ограничивает commercial use |

## Дополнительное чтение

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — performance targets, интеграция FastAPI
- [Mastra docs](https://mastra.ai/docs) — примитивы, server adapters, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — альтернатива stateful graph
- [Comet Opik](https://www.comet.com/site/products/opik/) — сравнения observability, cited by Mastra integrations
