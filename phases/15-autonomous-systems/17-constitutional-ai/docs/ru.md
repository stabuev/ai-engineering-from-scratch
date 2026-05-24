# Constitutional AI и переопределения правил

> Claude Constitution от Anthropic от 22 января 2026 года занимает 79 страниц и выпущена под CC0. Она переходит от rule-based к reason-based alignment и устанавливает четырехуровневую priority hierarchy: (1) safety and supporting human oversight, (2) ethics, (3) Anthropic guidelines, (4) helpfulness. Поведения делятся на hardcoded prohibitions (bioweapons uplift, CSAM), которые operators и users не могут override, и soft-coded defaults, которые operators могут настраивать в заданных bounds. Оригинал 2022 года (Bai et al.) обучал harmlessness через self-critique и RLAIF against a constitution. Честная оговорка: reason-based alignment полагается на то, что модель обобщает principles на unanticipated situations. Собственный participatory experiment Anthropic 2023 года показал ~50% divergence между public-sourced и corporate principles; версия 2026 года не учитывает эти результаты.

**Тип:** Изучение
**Языки:** Python (stdlib, четырехуровневый priority resolver)
**Предварительные требования:** Фаза 15 · 06 (Automated alignment research), Фаза 15 · 10 (Режимы разрешений)
**Время:** ~60 минут

## Проблема

Развернутый агент видит входы, которых его проектировщики никогда не видели. Ни один список правил не достаточно длинен, чтобы покрыть их. Ни один список правил не достаточно короток, чтобы быстро применяться под вычислительным давлением. Практический вопрос: как выровнять агента с принципами, которые выдерживают и long tail случаев, и быстрый inference?

Rule-based alignment (RBA): перечислить все запрещенное. Быстро проверяется, легко аудируется, невозможно поддерживать актуальным, часто приводит к over-refusal на близких аналогах, которых список не предвидел. Reason-based alignment (Claude Constitution 2026 года): закодировать принципы, позволить модели рассуждать. Масштабируется на unseen cases, сложнее аудируется, режим отказа — principle-misapplication, а не miss-the-rule.

Constitution 2026 года занимает явную среднюю позицию. Hardcoded prohibitions — вещи, неправильность которых не зависит от контекста (bioweapons uplift, CSAM), — это RBA: никогда, независимо от operator или user instruction. Все остальное reason-based внутри четырехуровневой hierarchy: safety and supporting human oversight сначала; ethics вторым; Anthropic-declared guidelines третьими; helpfulness последней. Operators могут настраивать defaults внутри soft-coded zone, но не могут трогать hardcoded prohibitions.

## Концепция

### Четырехуровневая priority hierarchy

1. **Safety and supporting human oversight.** Самый высокий уровень. Модель приоритизирует то, чтобы не подрывать способность людей и Anthropic осуществлять надзор и исправлять AI. Это не "be cautious"; это конкретно "do not act in ways that make human oversight harder."
2. **Ethics.** Честность, предотвращение вреда людям, отказ от обмана и манипуляций. Превосходит Anthropic guidelines, когда они конфликтуют.
3. **Anthropic guidelines.** Операционные нормы, которые Anthropic считает важными: product scope, interaction patterns, какие tools использовать и когда.
4. **Helpfulness.** Самый низкий. Быть максимально полезной в пределах более высоких priorities.

Когда уровни конфликтуют, выигрывает более высокий. Это та же форма, что Unix priorities или network QoS: framing должен давать предсказуемое разрешение, а не обязательно best-case behaviour по какой-то одной оси.

### Hardcoded prohibitions vs soft-coded defaults

**Hardcoded:**
- Bioweapons / CBRN uplift
- CSAM
- Атаки на критическую инфраструктуру
- Обман users о личности модели при прямом вопросе

Operator не может override это. User не может override это. Они enforced на уровне model weights, где возможно (RLHF / Constitutional AI training), и на inference layer, где нет.

**Soft-coded defaults (operator-adjustable):**
- Дефолтная длина ответа
- Topical scope (модель может refuse topics вне deployment operator)
- Стиль (formal vs casual)
- Паттерны использования инструментов

Operator adjustments происходят внутри declared bound. Operator не может удалить hardcoded prohibitions, переименовав их.

### Обучение CAI 2022 года

Оригинальная Constitutional AI (Bai et al., 2022) обучала harmlessness:

1. Сгенерировать ответы на набор prompts.
2. Попросить модель раскритиковать каждый ответ относительно constitution (явных principles).
3. Переписать ответ на основе critique.
4. Применить RLAIF (reinforcement learning from AI feedback) на revised pairs.

Результат: модель, которая refuses harmful requests с principled explanations, а не blanket refusals. Constitution 2026 года использует потомка этого обучения плюс дополнительное post-training по явной tier hierarchy.

### Что reason-based alignment ловит и пропускает

**Ловит:**
- Непредвиденные комбинации allowed primitives, где принцип ясно применим.
- Новые requests, являющиеся близкими analogs запрещенных.
- Social-engineering attacks, опирающиеся на "you didn't say X was disallowed."

**Пропускает:**
- Attacks, эксплуатирующие ambiguity принципов ("the user asked for this so helpfulness says yes").
- Scenarios, где два principles конфликтуют непредвиденным образом, а tier order ambiguous.
- Slow drift в principle interpretation за training cycles (reinterpretation).

### Participatory experiment 2023 года

Anthropic провела эксперимент 2023 года, сравнивая corporate-authored constitution с версией, generated via public input (~1,000 US respondents). Две версии совпали примерно по ~50% principles. Там, где они расходились, public-sourced version была более restrictive по некоторым вопросам (political-content handling) и менее restrictive по другим (self-disclosure of AI identity). Constitution 2026 года не учитывает public-sourced findings. Это задокументированное напряжение в подходе.

### Почему hardcoded prohibitions необходимы

Reason-based alignment alone не может закрыть tail. Атакующий, который может заставить модель принять premise (например, "we are a licensed bioweapons research lab"), часто может обойти принципы, зависящие от case reasoning. Hardcoded prohibitions не изгибаются под premise framing. Это "hard constitutional limit" из Урока 14 на alignment layer.

### Где Constitution находится в стеке

Constitution — это не kill switch из Урока 14. Она живет на model layer: чему weights модели обучены отдавать предпочтение. Kill switches и canary tokens живут на runtime layer: что runtime разрешает. Нужны оба. Runtime, который выполняет все неправильные actions, потому что model weights permissive, — это runtime problem. Модель, которая refuses все правильные actions, потому что runtime over-restrictive, — это runtime problem. Layers покрывают разные classes.

## Использование

`code/main.py` реализует минимальный четырехуровневый priority resolver. Resolver принимает proposed action и набор principle-evaluations (safety, ethics, guidelines, helpfulness) и возвращает action, refusal или modified action. Driver запускает небольшой case set: clear allow, clear disallow, hardcoded prohibition, ambiguous case across tiers.

## Практический результат

`outputs/skill-constitution-review.md` аудирует constitutional layer развертывания: что hardcoded, что soft-coded, где operator может adjust и действительно ли four-tier hierarchy является resolution order.

## Упражнения

1. Запустите `code/main.py`. Убедитесь, что hardcoded prohibition срабатывает даже когда helpfulness is high. Измените resolver так, чтобы weight helpfulness был выше ethics; наблюдайте failure mode.

2. Прочитайте Claude Constitution (public, 79 pages, CC0). Найдите один principle, который вы считаете under-specified. Напишите два абзаца, объясняющих конкретную ambiguity и предлагающих более tight formulation.

3. Спроектируйте набор soft-coded defaults для customer-support agent. Что operator adjust? Чего operator не может touch? Обоснуйте каждую boundary.

4. Прочитайте статью Bai et al. 2022 CAI. Опишите один случай, где Constitutional AI's critique-and-revise loop дала бы worse outcome, чем blanket rule. Определите class.

5. Participatory experiment Anthropic 2023 года показал ~50% divergence между public и corporate principles. Выберите одну category, где это важно для production deployment (например, political neutrality). Предложите design, позволяющий operators express their own values, пока hardcoded prohibitions remain untouched.

## Ключевые термины

| Термин | Как обычно говорят | Что это означает на самом деле |
|---|---|---|
| Constitutional AI | "Метод alignment от Anthropic" | Self-critique + RLAIF against a written constitution |
| Reason-based alignment | "Принципы, а не правила" | Модель рассуждает о principles для обработки unseen cases |
| Hardcoded prohibition | "Никогда не делай X" | Rule-based prohibition, который не может override ни operator, ни user |
| Soft-coded default | "Настраивается operator" | Behaviour внутри declared bound, управляется operator |
| Four-tier hierarchy | "Порядок приоритетов" | safety > ethics > guidelines > helpfulness |
| RLAIF | "RL на AI feedback" | RL, где reward приходит из model-generated critiques |
| Participatory constitution | "Принципы от общественности" | Эксперимент Anthropic 2023 года; ~50% divergence from corporate |
| Principle drift | "Сдвиг интерпретации" | Slow change in how the model reads a fixed principle text |

## Дополнительное чтение

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79-страничный документ CC0.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — оригинальная работа 2022 года.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — participatory experiment с участием общественности.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — где Constitution находится в RSP stack.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — роль Constitution в long-horizon deployments.
