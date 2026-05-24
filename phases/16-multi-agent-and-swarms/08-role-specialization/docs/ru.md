# Ролевая специализация — Planner, Critic, Executor, Verifier

> Самая распространенная multi-agent декомпозиция в 2026 году: один агент планирует, один исполняет, один критикует или проверяет. MetaGPT (arXiv:2308.00352) формализует это как SOPs, закодированные в role prompts — Product Manager, Architect, Project Manager, Engineer, QA Engineer — следуя `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) связывает designer, programmer, reviewer, tester через "chat chain" с "communicative dehallucination" (агенты явно запрашивают недостающие детали). Verifier является несущим элементом: Cemri et al. (MAST, arXiv:2503.13657) показывают, что каждый multi-agent failure можно проследить до отсутствующей или сломанной verification. PwC сообщила о 7× росте accuracy (10% → 70%) от structured validation loops в CrewAI.

**Тип:** Изучение + сборка
**Языки:** Python (stdlib)
**Пререквизиты:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Время:** ~60 минут

## Проблема

Generic multi-agent systems производят generic output. Три кодера в group chat пишут три варианта одного и того же посредственного кода. Можно добавить больше агентов, больше раундов и все равно не перейти порог качества.

Исправление - не больше агентов, а *разные* агенты. Назначьте разные роли. Дайте critic инструменты, которых нет у planner. Дайте verifier объективный test suite. Теперь у системы есть внутреннее несогласие с grounded correction, а не просто parallel guessing.

## Концепция

### Четыре канонические роли

**Planner.** Читает цель, выдает список шагов или spec. Инструменты: knowledge retrieval, docs. Выход: structured plan.

**Executor.** Читает по одному шагу плана и создает artifact. Инструменты: реальные рабочие инструменты (code compiler, shell, API client). Выход: artifact.

**Critic.** Читает output executor относительно intent planner. Инструменты: read-only доступ к artifact, static analysis. Выход: accept/reject с причинами.

**Verifier.** Читает artifact и запускает deterministic check. Инструменты: test runner, type checker, schema validator. Выход: pass/fail с evidence.

Critic субъективен, opinionated, часто LLM-based. Verifier объективен, deterministic, часто code-based. Это не одна и та же роль.

### SOP-паттерн MetaGPT

MetaGPT (arXiv:2308.00352) кодирует software engineering SOPs как role prompts:

- **Product Manager** пишет PRD.
- **Architect** создает system design.
- **Project Manager** разбивает задачи.
- **Engineer** реализует.
- **QA Engineer** запускает tests.

У каждой роли строгая input/output schema. Role prompt говорит, чем роль *является* и что она *должна произвести*. Формулировка `Code = SOP(Team)` - deterministic SOPs превращают команду LLMs в предсказуемый pipeline.

### Communicative dehallucination в ChatDev

ChatDev добавляет ключевой ход: когда executor нужна конкретная деталь, которой не было в плане, он явно спрашивает designer перед продолжением. Это предотвращает классический отказ LLM - правдоподобно выдумать деталь.

Реализация: role prompt включает "when you need specific information you were not given, ask the relevant role by name before producing output."

### Почему verifier важнее всего

Cemri et al. (MAST) проследили 1642 multi-agent execution failures. 21.3% были verification gaps - система отправила ответ, который никто не проверил. Оставшиеся 79% часто восходят к "была проверка, которая тихо провалилась или не была запущена." Verification - несущая роль.

PwC сообщила (CrewAI deployments, 2025), что добавление structured validation loop подняло accuracy с 10% до 70%. 7× gain от одной роли.

### Critic vs verifier

- Critic - это LLM, reviewing artifact на качество. Субъективен. Его можно обмануть правдоподобной прозой.
- Verifier - это deterministic program, running on the artifact. Объективен. Дает pass/fail с evidence.

Используйте оба. Critic ловит taste issues, которые verifier не может артикулировать. Verifier ловит bugs, которые critic не видит, потому что они проявляются только в runtime.

### Антипаттерн

Каждая роль в вашей системе - LLM, и output каждой роли - "looks good to me." Классический режим отказа MAST. Добавьте хотя бы одного verifier, чей pass/fail определяется code, а не LLM.

### Соответствия фреймворкам

- **CrewAI** — `Agent(role, goal, backstory)` является учебниковой поверхностью specialization.
- **LangGraph** — nodes могут иметь specialized prompts; edges enforce the pipeline.
- **AutoGen** — role-specific ConversableAgents с one-word names в GroupChat.
- **OpenAI Agents SDK** — handoff tools между role-specialized Agents.

## Соберите это

`code/main.py` реализует 4-role pipeline, который строит простую Python function:

- **Planner** создает spec.
- **Executor** генерирует code string.
- **Critic** (LLM-simulated) отмечает очевидные issues.
- **Verifier** запускает generated code в sandbox (`exec`) на test case.

Демо запускается дважды: один раз executor создает корректный code (critic + verifier оба pass), второй раз executor создает off-spec code (critic пропускает bug, потому что он выглядит plausible, verifier ловит его, потому что test fails).

Запуск:

```
python3 code/main.py
```

## Используйте это

`outputs/skill-role-designer.md` принимает задачу и выдает role roster (3-5 ролей), input/output schema для каждой роли и verifier check. Используйте это перед wiring agents into a framework.

## Доведите до production

Чеклист:

- **Минимум один deterministic verifier.** Никогда не all-LLM.
- **Явная I/O schema для каждой роли.** Planner возвращает spec, а не prose; executor читает эту schema.
- **Communicative dehallucination.** Executor должен спрашивать planner, когда информации не хватает; никогда не выдумывать ее.
- **Порядок critic/verifier.** Запускайте critic первым (cheap, ловит design issues), verifier вторым (slow, ловит bugs).
- **Loop budget.** Max 2 critic-executor revision rounds перед escalation to human.

## Упражнения

1. Запустите `code/main.py` и посмотрите, как verifier ловит bug, который critic пропустил. Добавьте static-analysis check (подсчет occurrences of `return`) как дополнительный verifier. Что он ловит из того, что runtime test пропускает?
2. Добавьте 5-ю роль: "requirements analyst", которая переводит user wish в planner-ready spec. Какие communicative dehallucination requests должны подниматься к ней?
3. Прочитайте MetaGPT Section 3 ("Agents"). Перечислите input/output schema каждой из 5 ролей MetaGPT.
4. Прочитайте ChatDev's chat-chain diagram (arXiv:2307.07924 Figure 3). Определите, где communicative dehallucination разрывает loop, который иначе был бы infinite.
5. 7× accuracy gain PwC пришел из verification loops. Предположите три задачи, где добавление verifier не поможет - где deterministic checking of correctness невозможна или чрезмерно дорога.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Role specialization | "Разные агенты, разные работы" | Разные system prompts, настроенные под роли planner/executor/critic/verifier. |
| SOP pattern | "Закодированная standard operating procedure" | Рамка MetaGPT: строгие I/O schemas по ролям превращают команду в pipeline. |
| Communicative dehallucination | "Спроси перед тем, как выдумывать" | Паттерн ChatDev: executor спрашивает planner, когда деталь отсутствует, вместо того чтобы ее выдумывать. |
| Critic | "LLM reviewer" | Субъективный, opinionated reviewer. Ловит taste issues. Может быть обманут plausible prose. |
| Verifier | "Deterministic check" | Code-based pass/fail. Test runner, type checker, schema validator. Его нельзя обмануть. |
| Verification gap | "Никто не проверил" | 21.3% failures в MAST. Ответ отправлен без проверки, которая поймала бы bug. |
| Revision loop | "Critic отправляет назад" | Rejection от critic запускает executor re-run с feedback. Нужен budget. |
| All-LLM anti-pattern | "Looks good to me" | Каждая роль - LLM, нет deterministic check. Классический failure MAST. |

## Дополнительное чтение

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — базовая статья по SOP-as-role-prompt
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — таксономия MAST; verification gaps составляют 21.3% failures
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — production-поверхность для specification of roles
