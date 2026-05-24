# Ландшафт автономных coding agents (2026)

> SWE-bench Verified вырос с 4% до 80.9% менее чем за три года. Один и тот же Claude Sonnet 4.5 набрал 43.2% на SWE-agent v1 и 59.8% на автономном Cline — каркас вокруг модели теперь важен не меньше, чем сама модель. OpenHands (ранее OpenDevin) — самая активная платформа с MIT-лицензией, а ее цикл CodeAct выполняет Python-действия напрямую в sandbox вместо JSON-вызовов инструментов. Заголовочные числа скрывают методологическую проблему: 161 из 500 задач SWE-bench Verified требуют изменения всего в 1–2 строки, а SWE-bench Pro (задачи 10+ строк) находится на уровне 23–59% для тех же frontier models.

**Тип:** Обучение
**Языки:** Python (stdlib, сравнение CodeAct и JSON-вызовов инструментов)
**Предварительные требования:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Время:** ~45 минут

## Проблема

«Какой coding agent лучший?» — неправильный вопрос. Правильный вопрос: какую сквозную надежность я получу на распределении задач, совпадающем с моей работой, и с каркасом, который буду запускать в production?

С 2022 по 2026 год область поняла, что каркас — слой retrieval, planner, sandbox, цикл edit-verify, формат обратной связи — является несущей конструкцией. Claude Sonnet 4.5 на SWE-agent v1 набрал 43.2% на SWE-bench Verified; та же модель внутри автономного каркаса Cline набрала 59.8%. Разница 16.6 абсолютных пункта, те же веса. Базовая модель — компонент; цикл — продукт.

Сопутствующая проблема в том, что насыщение бенчмарков скрывает регрессии. SWE-bench Verified близок к насыщению, а хвост легких задач (161 из 500 задач, требующих ≤2 строк) подтягивает верхние результаты вверх. Реальное качество лучше измерять на распределениях вроде SWE-bench Pro (изменения 10+ строк), где те же лидеры все еще находятся на 23–59%.

## Концепция

### SWE-bench в одном абзаце

SWE-bench (Jimenez et al.) берет реальные GitHub issues с ground-truth patches и просит агента создать patch, после которого тестовый набор проходит. SWE-bench Verified (OpenAI, 2024) — human-curated подмножество из 500 задач, из которого удалены неоднозначные и сломанные задачи. SWE-bench Pro — более сложный преемник: задачи, требующие 10+ строк изменений, где текущие frontier agents находятся на уровне 23–59%.

### Что на самом деле показывает кривая 2022 → 2026

- **2022**: исследовательские модели на ~4% на raw SWE-bench.
- **2024**: GPT-4 + Devin-style scaffolding на ~14%; SWE-agent на ~12%.
- **2025**: Claude 3.5/3.7 Sonnet внутри Aider и SWE-agent выходят в диапазон 40–55%.
- **2026**: Claude Sonnet 4.5 и frontier competitors на 70–80%+ на SWE-bench Verified. Leaderboard Epoch AI отслеживает это вживую.

Наклон появился из трех накапливающихся источников: лучшие базовые модели, лучший scaffolding (CodeAct, reflection, verifier loops) и лучшие бенчмарки (Verified удаляет шум).

### CodeAct против JSON-вызовов инструментов

OpenHands (All-Hands-AI, arXiv:2407.16741, ранее OpenDevin) сделал конкретную архитектурную ставку: вместо того чтобы модель выдавала JSON-вызовы инструментов, которые host декодирует и выполняет, модель выдает Python-код, а Jupyter-style kernel запускает его в sandbox. Агент может обходить файлы, цеплять инструменты и ловить собственные исключения внутри одного действия.

Компромисс:

- **JSON tool calls**: каждое действие — один turn; легко аудитить; ограниченная композиционность; безопасно по умолчанию, потому что каждый вызов проходит через явный validator.
- **CodeAct**: одно действие может быть целой программой; композиционен; требует усиленного sandbox (OpenHands использует Docker isolation); режимы отказа включают все, что допускает runtime sandbox.

Обе архитектуры используются в production. CodeAct доминирует в открытых платформах (OpenHands, smolagents). JSON tool calls остаются доминирующими в управляемых сервисах (Anthropic Managed Agents, OpenAI Assistants), где провайдер контролирует executor.

### Каркасы в ландшафте 2026 года

| Каркас | Лицензия | Модель выполнения | Примечательное свойство |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct in Docker | Самая активная открытая платформа; event-stream можно воспроизводить |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | Первый end-to-end SWE-bench scaffold |
| Aider | Apache-2 | edit-via-diff in local repo | Минимальный scaffold, сильная стабильность регрессий |
| Cline | Apache-2 | VS Code agent with tool policy | Самый высоко оцениваемый открытый scaffold на Sonnet 4.5 |
| Devin (Cognition) | Proprietary | Managed VM + planner | Первая продуктовая категория «AI software engineer» |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 подробно разбирает agent loop |

### Почему scaffolding доминирует

Coding run — это long-horizon trajectory (Lesson 1). Надежность накапливается по шагам. Три места, где scaffolding дает пункты:

1. **Retrieval**: найти правильные файлы для чтения — тихое узкое место. ACI от SWE-agent, file-index OpenHands и repo-map Aider атакуют именно это.
2. **Verifier loop**: запуск тестов, чтение stack traces и повторная попытка дают дельту 10+ пунктов на SWE-bench.
3. **Failure containment**: sandbox, который откатывает при ошибке, предотвращает накопление ущерба. Одна и та же модель с verifier loop и без него выглядит как два разных продукта.

### Насыщение бенчмарков и реальное распределение

Авторы OpenHands и Epoch AI оба отмечают, что у SWE-bench Verified есть легкий хвост: 161 из 500 задач требуют всего 1–2 строки изменений. Высокие результаты частично порождены этим хвостом. SWE-bench Pro ограничивается изменениями 10+ строк и дает результаты в диапазоне 23–59% даже для frontier systems. Ваше production-распределение почти наверняка ближе к Pro, чем к Verified.

Следствие для выбора агента: запустите Pro-подобное подмножество собственного bug backlog. Важен результат на задачах, репрезентативных для того, что вы поставляете.

## Использование

`code/main.py` сравнивает два игрушечных агентных каркаса на фиксированном распределении мини-задач:

1. Каркас **JSON tool-call**, который делает одно действие за turn.
2. Каркас **CodeAct**, который может выдавать небольшой Python snippet за действие.

Оба используют stub «model» (детерминированные правила), поэтому сравнение изолирует каркас от качества модели. Вывод показывает, что каркас CodeAct решает больше задач за меньшее число turns ценой большего blast radius на действие.

## Результат

`outputs/skill-scaffold-audit.md` помогает аудитить предлагаемый каркас coding-agent перед внедрением: качество retrieval, наличие verifier, изоляцию sandbox и соответствие benchmark-to-distribution.

## Упражнения

1. Запустите `code/main.py`. Сколько turns занимает каждый scaffold на одном и том же наборе задач? Каков per-action blast radius у каждого?

2. Прочитайте статью OpenHands (arXiv:2407.16741). В статье утверждается, что CodeAct превосходит JSON tool calls на сложных задачах. Определите один режим отказа, который признает статья, и напишите одно предложение о том, когда этот режим доминировал бы в production.

3. Выберите одну задачу из вашего bug backlog, которая потребовала бы 10+ строк изменений в двух файлах. Оцените end-to-end вероятность успеха для frontier model при (a) JSON tool calls и (b) CodeAct. Обоснуйте разрыв.

4. В SWE-bench Verified есть 161 single-file задача на 1–2 строки. Сконструируйте метрику, которая исключает их. Как перемешается leaderboard?

5. Прочитайте "Introducing SWE-bench Verified" (OpenAI). Объясните конкретную методологию, использованную для удаления неоднозначных задач, и назовите одну категорию, которую curation пропустил бы.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|---|---|---|
| SWE-bench | «Бенчмарк программирования» | Реальные GitHub issues с ground-truth patches и test suites |
| SWE-bench Verified | «Очищенное подмножество» | 500 human-curated задач, присутствует легкий хвост |
| SWE-bench Pro | «Более сложное подмножество» | Изменения 10+ строк; frontier находится на 23–59% |
| CodeAct | «Code-as-action» | Агент выдает Python; Jupyter-style kernel выполняет в sandbox |
| JSON tool call | «Function calling» | Каждое действие — структурированный JSON payload, валидируемый перед выполнением |
| Scaffold | «Agent framework» | Retrieval + planner + executor + verifier loop вокруг базовой модели |
| ACI (Agent-Computer Interface) | «Формат SWE-agent» | Набор команд, спроектированный для эргономики LLM, а не human shells |
| Verifier loop | «Test-and-retry» | Запуск тестов, чтение вывода, правка patch; главный прирост надежности не от модели |

## Дополнительное чтение

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — исходный бенчмарк и методология.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — как было построено curated subset.
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — архитектура CodeAct и event-stream design.
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — результаты, отслеживаемые вживую.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — рамка надежности long-horizon coding-agent.
