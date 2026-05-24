# Self-Refine и CRITIC: итеративное улучшение результата

> Self-Refine (Madaan et al., 2023) использует одну LLM в трёх ролях — generate, feedback, refine — в цикле. Средний прирост: +20 absolute на 7 tasks. CRITIC (Gou et al., 2023) усиливает шаг feedback, маршрутизируя verification через внешние инструменты. В 2026 году этот паттерн поставляется в каждом фреймворке как "evaluator-optimizer" (Anthropic) или цикл guardrail (OpenAI Agents SDK).

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 03 (Reflexion)
**Время:** ~60 минут

## Цели обучения

- Сформулировать три промпта Self-Refine (generate, feedback, refine) и объяснить, почему history важна для refine prompt.
- Объяснить критический insight CRITIC: LLM ненадёжны в self-verification без внешнего grounding.
- Реализовать цикл Self-Refine на stdlib с history и optional external verifier.
- Сопоставить этот паттерн с workflow Anthropic "evaluator-optimizer" и output guardrails в OpenAI Agents SDK.

## Проблема

Агент выдаёт ответ, который почти правильный. Возможно, в строке кода syntax error. Возможно, summary слишком длинное. Возможно, план пропускает edge case. Нужно, чтобы агент раскритиковал собственный результат, а затем исправил его.

Self-Refine показывает, что это работает с одной моделью, без training data, без RL. Но есть подвох: LLM плохо проверяют сами себя на hard facts. CRITIC предлагает исправление — направить шаг verify через внешние инструменты (search, code interpreter, calculator, test runner).

Вместе эти две статьи определяют default 2026 года для итеративного улучшения: generate, verify (внешне, когда возможно), refine, stop when the verifier passes.

## Концепция

### Self-Refine (Madaan et al., NeurIPS 2023)

Одна LLM, три роли:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

Ключевая деталь: `refine` видит полную history — все предыдущие outputs и critiques, — поэтому не повторяет ошибки. Статья делает ablation: убрать history — и качество резко падает.

Главный результат: среднее улучшение +20 absolute по 7 задачам (математика, код, акронимы, диалог), включая GPT-4. Без training, без внешних инструментов, одна модель.

### CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024)

Слабость Self-Refine: шаг feedback — это LLM, оценивающая саму себя. Для factual claims это ненадёжно (галлюцинация часто выглядит убедительно для модели, которая её произвела). CRITIC заменяет `feedback(task, output)` на `verify(task, output, tools)`, где `tools` включает:

- Search engine для factual claims.
- Code interpreter для проверки корректности кода.
- Calculator для arithmetic.
- Domain-specific verifiers (unit tests, type checkers, linters), то есть доменные проверяющие.

Verifier создаёт структурированную critique, заземлённую на результатах инструментов. Затем refiner учитывает эту critique.

Главный результат: CRITIC превосходит Self-Refine на factual tasks, потому что critique заземлена. На задачах без внешних verifiers (creative writing, formatting) CRITIC сводится к Self-Refine.

### Условие остановки

Две распространённые формы:

1. **Verifier passes.** Внешний тест возвращает success. Предпочтительно, когда доступно (unit tests, type checker, guardrail assertion).
2. **No feedback issued.** Модель говорит "the output is fine." Дешевле, но ненадёжно; сочетайте с лимитом итераций.

Default 2026 года: комбинировать. "Stop if verifier passes OR model says fine AND iterations >= 2 OR iterations >= max_iterations."

### Evaluator-Optimizer (Anthropic, 2024)

Пост Anthropic Dec 2024 называет это одним из пяти workflow patterns. Две роли:

- Evaluator: оценивает результат и создаёт critique.
- Optimizer: перерабатывает результат с учётом critique.

Цикл продолжается, пока evaluator не пропустит результат. Это Self-Refine/CRITIC во framing Anthropic. Критически важная инженерная деталь от Anthropic: промпты evaluator и optimizer должны существенно отличаться, чтобы модель не ставила rubber-stamp.

### OpenAI Agents SDK output guardrails

OpenAI Agents SDK поставляет этот паттерн как "output guardrails." Guardrail — это validator, который запускается на final output агента. Если guardrail trips (raises `OutputGuardrailTripwireTriggered`), output rejected, и agent can retry. Guardrails могут вызывать tools (в стиле CRITIC) или быть pure functions (в стиле Self-Refine).

### Ловушки 2026 года

- **Rubber-stamp loops.** Одна и та же модель выполняет generation и critique с тем же стилем промпта и сходится к "looks good to me." Используйте структурно разные промпты или более дешёвую маленькую модель для critique.
- **Over-refinement.** Каждый проход refine добавляет latency и токены. Бюджетируйте 1-3 прохода; после этого escalation to human review.
- **CRITIC on trivial tasks.** Если external verifier отсутствует, CRITIC вырождается в Self-Refine; не платите latency за stub verifier.

## Соберите это

`code/main.py` реализует Self-Refine и CRITIC на игрушечной задаче: создать короткий список bullet points по теме. Verifier проверяет формат (3 bullets, each under 60 chars). CRITIC добавляет внешний "fact verifier", который штрафует известные галлюцинации.

Компоненты:

- `generate` — scripted producer.
- `feedback` — self-critique в стиле LLM.
- `verify_external` — grounded verifier в стиле CRITIC.
- `refine` — переписывает output с учётом history.
- Stop condition — verifier passes или максимум 4 итерации.

Запустите:

```
python3 code/main.py
```

Сравните запуски Self-Refine vs CRITIC. CRITIC ловит factual error, которую Self-Refine пропустил, потому что у external verifier есть grounding, которого нет у self-critic.

## Используйте это

Evaluator-optimizer Anthropic — это этот паттерн на языке, удобном для Claude. Output guardrails в OpenAI Agents SDK имеют форму CRITIC (guardrails могут вызывать tools). LangGraph поставляет reflection node, который читается как Self-Refine. Google Gemini 2.5 Computer Use добавляет per-step safety evaluator, который является вариантом CRITIC: каждое действие проверяется перед commit.

## Отгрузите это

`outputs/skill-refine-loop.md` конфигурирует цикл evaluator-optimizer по форме задачи, доступности verifier и бюджету итераций. Выдаёт промпты для generator, evaluator/verifier и optimizer, плюс stop policy.

## Упражнения

1. Запустите toy с max_iterations=1. CRITIC всё ещё помогает?
2. Замените external verifier на шумный (random 30% false positives). Что делает loop? Это реальность 2026 года для большинства guardrail stacks.
3. Реализуйте вариант "generator-critic on different models": большая модель генерирует, маленькая критикует. Он превосходит same-model?
4. Прочитайте CRITIC Section 3 (arXiv:2305.11738 v4). Назовите три категории verification-tool и дайте пример для каждой.
5. Сопоставьте `output_guardrails` OpenAI Agents SDK с ролью verifier в CRITIC. Что SDK делает неправильно, а что правильно?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Self-Refine | "LLM that fixes itself" | Цикл generate -> feedback -> refine в одной модели, с history |
| CRITIC | "Tool-grounded verification" | Замена feedback внешним verifier (search, code, calc, tests) |
| Evaluator-Optimizer | "Anthropic workflow pattern" | Две роли — evaluator оценивает, optimizer перерабатывает — цикл до сходимости |
| Output guardrail | "Post-hoc check" | Validator OpenAI Agents SDK, который запускается после того, как agent produces output |
| Verify step | "Critique phase" | Несущее решение: grounded or self-rated |
| Refine history | "What the model already tried" | Предыдущие outputs + critiques, добавленные в refine prompt; убрать — и качество падает |
| Rubber-stamp loop | "Self-agreement failure" | Same-prompt critique возвращает "looks good"; исправляется структурно разными промптами |
| Stop condition | "Convergence test" | Verifier passes OR no feedback AND iteration cap; не используйте одно условие |

## Дополнительное чтение

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — каноническая статья
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — верификация, заземлённая на инструменты
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — workflow pattern evaluator-optimizer
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — output guardrails как verifiers в форме CRITIC
