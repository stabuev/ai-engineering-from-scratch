# Automated Alignment Research (Anthropic AAR)

> Anthropic запустила параллельные команды Claude Opus 4.6 Autonomous Alignment Researchers в независимых sandboxes, координирующихся через shared forum, логи которого живут вне любого sandbox (поэтому agents не могут удалить собственные records). На задаче weak-to-strong training AARs превзошли исследователей-людей. Собственное summary Anthropic отмечает, что prescribed workflows часто ограничивают гибкость AAR и ухудшают производительность. Автоматизация alignment research — это шаг сжатия, который приближает timeline к тем самым misalignment risks, которые RSP должен обнаруживать.

**Тип:** Изучение
**Языки:** Python (stdlib, симулятор parallel-research-forum)
**Предварительные требования:** Фаза 15 · 05 (AI Scientist v2), Фаза 15 · 04 (DGM)
**Время:** ~60 минут

## Проблема

Alignment research дорог по времени исследователей-людей. Задачи вроде scalable oversight, reward specification или weak-to-strong training требуют экспериментов, где одна итерация занимает недели. По мере роста frontier capabilities нагрузка alignment растет быстрее, чем предложение квалифицированных исследователей.

Automated Alignment Research (AAR) задает вопрос, могут ли те же frontier models, чьи возможности обгоняют alignment, помогать закрывать разрыв. Отчет Anthropic 2026 года об AAR-run weak-to-strong-training study (alignment.anthropic.com/2026/automated-w2s-researcher/) — один из первых публичных результатов развернутой системы такого класса.

Результат действительно положительный: AARs решили исследовательскую задачу лучше human baseline. Этот результат также концентрирует в одной системе все опасения, которые развила эта фаза. Если alignment research можно автоматизировать, то можно автоматизировать и части, которые компрометируют safeguards. Пороги RSP для AI R&D capability написаны с учетом этого цикла.

## Концепция

### Архитектура (как публично описана)

- **Параллельные агенты.** Несколько экземпляров Claude Opus 4.6 работают как AARs, каждый в независимом sandbox. Sandboxes ограничивают filesystem, network и process surface.
- **Shared forum.** Findings публикуются в shared forum, backend хранения которого находится вне каждого sandbox. Agents могут читать forum; agents не могут удалять или редактировать forum records из своего sandbox.
- **Распределение задач.** Human-designed task queue назначает исследовательские задачи AARs. AARs предлагают decompositions, запускают experiments, пишут findings.
- **Фокус на weak-to-strong training.** Опубликованное исследование рассматривало weak-to-strong training — использование более слабой supervisor model для обучения более сильной student model. AARs превзошли исследователей-людей на этой задаче.

### Почему out-of-sandbox log важен

Свойство целостности log — это разница между контролируемым экспериментом и потенциально скомпрометированным. Если бы AARs могли редактировать собственные logs, они могли бы скрывать неудачные или сомнительные experiments. Anthropic прямо отмечает это: такое свойство является необходимым условием доверия к research output, и его нетривиально проверить.

Аналог в инфраструктуре — append-only logs с write-through во внешнее хранилище (например, S3 object-lock, immutable Datadog logs). У агента нет credential для изменения прошлых records. Это тот же паттерн, который corporate security teams 2026 года используют для audit trails, устойчивых к ransomware.

### Tradeoff prescribed-workflow

Отчет Anthropic отмечает, что prescribed workflows часто ограничивают гибкость AAR и ухудшают производительность. Иными словами: если human designer пишет research plan step-by-step, AARs справляются хуже, чем при получении high-level goals и свободы decomposition. Это согласуется с результатами в agentic research (LangGraph, Microsoft's Agent Framework) — overspecified scaffolds вредят способным agents.

Safety implication тоньше. Гибкость, улучшающая производительность AAR, является также гибкостью, которая допускает unsanctioned decomposition, novel tool use и goal-drift. Tradeoff — это решение в design space: сколько гибкости покупает достаточно capability, чтобы превзойти людей, не отдавая агенту пространство для обобщения целей off-distribution?

### Риск сжатия

RSP v3.0 (Урок 19) вводит порог AI R&D capability: способность полностью автоматизировать AI R&D pipeline по competitive cost относительно human + AI tools. FSF v3 DeepMind включает аналогичный уровень ML R&D autonomy. Обе рамки трактуют этот порог как trigger для elevated controls.

AAR находится в одном шаге от порога: он автоматизирует часть pipeline (alignment research на конкретных, хорошо ограниченных задачах), но не end-to-end capability development loop. Вопрос timeline — насколько быстро закрывается разрыв.

Compressed timelines — это concern о compounding failure. Если alignment research и capability research накапливают прогресс с похожей скоростью, поверхность misalignment risk растет как минимум так же быстро, как capability. Если capability накапливается быстрее (исторический тренд), разрыв расширяется. Это аргумент в пользу AAR как qualified good: каждый дополнительный alignment result уменьшает разрыв тогда и только тогда, когда research process заслуживает доверия.

### Что AAR не заменяет

Исследователи-люди задают task queue, проверяют results и удерживают constitutional authority. AARs ускоряют середину pipeline, а не его края. Опубликованные outputs Anthropic включают вклад AAR и judgement исследователей-людей о том, что публиковать, что отзывать и что дорабатывать.

Это соответствует паттерну propose-then-commit из Урока 15, примененному к самому research: AARs propose; humans commit.

## Использование

`code/main.py` симулирует multi-AAR research forum. Три независимых "research agents" работают параллельно, публикуя findings в append-only forum с опциональными tamper attempts.

Наблюдайте:

- Как out-of-sandbox logging ловит log-tampering attempts.
- Как task allocation со free decomposition превосходит fixed workflows на более сложных задачах.
- Как один compromised AAR может загрязнить signal forum, если log не tamper-evident.

## Практический результат

`outputs/skill-aar-deployment-review.md` — pre-deployment review для automated-alignment-research pipeline.

## Упражнения

1. Запустите `code/main.py`. Сравните настройки "fixed-workflow" и "free-decomposition". Всегда ли free decomposition выигрывает, или есть классы задач, где fixed workflow конкурентен?

2. Измените simulator так, чтобы один agent пытался log tampering. Подтвердите, что append-only log это обнаруживает. Напишите один абзац, точно описывающий, как выглядит detection signature в log.

3. Прочитайте отчет Anthropic weak-to-strong AAR. Определите конкретную sub-task, в которой AARs превзошли исследователей-людей. Что сделало ее пригодной для automation?

4. Спроектируйте policy распределения task queue, которая балансирует гибкость AAR (лучшие results) против ограничений prescribed-workflow (легче audit). Опишите, как вы провели бы A/B test этих двух вариантов.

5. Прочитайте порог AI R&D-4 в RSP v3.0. В одном абзаце опишите, что, по вашему мнению, пересекло бы его, но чего AAR сейчас не делает.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|---|---|---|
| AAR | "Automated Alignment Researcher" | Экземпляр Claude Opus 4.6, автономно работающий над alignment problems |
| Weak-to-strong training | "Обучение более сильной модели слабым supervisor" | Классический scalable-oversight benchmark, на котором AARs превзошли людей |
| Shared forum | "Где agents публикуют findings" | Append-only, out-of-sandbox storage |
| Out-of-sandbox log | "Agent не может редактировать собственную запись" | Tamper-evident write-through во внешнее хранилище |
| Prescribed workflow | "Пошаговый план от human designer" | Ограничивает AAR; часто ухудшает performance по сравнению со free decomposition |
| Free decomposition | "Агент сам разбивает задачу" | Более capable, труднее audit |
| AI R&D threshold | "Уровень capability в RSP/FSF" | Полная автоматизация R&D pipeline по competitive cost |
| Compressed timeline | "Alignment vs capability race" | Если capability compounding быстрее alignment, misalignment risk растет |

## Дополнительное чтение

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — primary source.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — framing порога AI R&D.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — более широкий framing agent-autonomy.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — уровни ML R&D autonomy, параллельные RSP.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — базовая проблема, которую атаковали AARs.
