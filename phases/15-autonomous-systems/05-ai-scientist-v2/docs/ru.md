# AI Scientist v2 — автономные исследования уровня workshop

> AI Scientist v2 от Sakana (Yamada et al., arXiv:2504.08066) выполняет полный исследовательский цикл: hypothesis, code, experiments, figures, writeup, submission. Это первая система, у которой сгенерированная статья прошла peer review на ICLR 2025 workshop. Независимая оценка (Beel et al.) обнаружила, что 42% экспериментов провалились из-за coding errors, а literature review часто ошибочно помечал известные концепции как novel. Собственная документация Sakana предупреждает, что codebase выполняет код, написанный LLM, и рекомендует Docker isolation. Обе половины этой картины и есть суть.

**Тип:** Изучение
**Языки:** Python (stdlib, игрушечная research-loop state-machine)
**Предварительные требования:** Фаза 15 · 03 (AlphaEvolve), Фаза 15 · 04 (DGM)
**Время:** ~60 минут

## Цели обучения

- Описывать полный исследовательский цикл, который автоматизирует AI Scientist v2 (гипотеза → код → эксперименты → текст → подача).
- Определять, что позволило ему первым пройти планку рецензирования, и оговорки.
- Рассуждать, где автоматизированная наука пока проваливается.

## Проблема

Исследование — открытая задача. В отличие от algorithmic search в AlphaEvolve или benchmark-bounded self-modification в DGM, у исследовательского результата нет машинно-проверяемого критерия корректности. Статью оценивают рецензенты, а не unit tests. Это делает цикл труднее замкнуть — и более ценным, если его удается замкнуть, потому что research — место, где живет compounding progress.

AI Scientist v1 (Sakana, 2024) замкнул цикл, начиная с шаблонов, написанных людьми. LLM заполняла эксперименты внутри фиксированного scaffolding. AI Scientist v2 (Yamada et al., 2025) убирает требование шаблонов, используя agentic tree search с циклом critique от vision-language model. Система генерирует идеи, реализует эксперименты, производит figures, пишет статью и итерирует по reviewer feedback.

Вердикт peer review: одна статья, сгенерированная v2, была принята на ICLR 2025 workshop (с disclosure). Вердикт независимой оценки: система далеко не надежна. Верны оба утверждения.

## Концепция

### Архитектура

1. **Генерация идей.** LLM предлагает исследовательские идеи с учетом темы и предыдущей литературы. v1 использовала templates; v2 использует agentic search по пространству hypotheses.
2. **Проверка новизны.** Шаг literature retrieval проверяет, была ли идея опубликована. Именно на этом шаге оценка Beel et al. обнаружила mislabeling — известные методы часто классифицировались как novel.
3. **План эксперимента.** Агент формирует experimental protocol и пишет код.
4. **Выполнение.** Код запускается в sandbox. Отказы возвращаются в retry loop. В измерениях Beel et al. 42% экспериментов провалились из-за coding errors на этом этапе.
5. **Генерация figures.** Vision-language model читает сгенерированные figures и переписывает их для ясности. Это было ключевым техническим добавлением v2.
6. **Writeup.** LLM пишет черновик статьи и итерирует с internal reviewer.
7. **Опционально: submission.** Статья отправляется на venue.

### Что означает результат workshop acceptance

Одна статья, сгенерированная v2, прошла peer review на ICLR 2025 workshop. Авторы раскрыли происхождение статьи program committee. Принятие — это datapoint; это не основание утверждать, что система полноценно "занимается исследованиями".

Важный контекст: workshop papers имеют более низкую планку, чем main-conference papers. Peer review шумный; небольшая доля submissions принимается в любой конкретный день. Один успех — proof of concept, а не утверждение о reliability. Статья Nature 2026 документирует end-to-end loop и сама была написана в соавторстве с исследователями-людьми; это не "система написала статью в Nature".

### Что нашла независимая оценка

Beel et al. (arXiv:2502.14297) провели внешнюю оценку. Главные находки:

- **Провалы экспериментов.** 42% экспериментов провалились из-за coding errors (bad imports, shape mismatches, undefined variables). Retry loop поймал часть, но не все.
- **Ошибочная маркировка новизны.** Шаг literature-retrieval часто помечал известные концепции как novel. Это исследовательский аналог hallucination.
- **Разрыв качества презентации.** Vision-language figure critique производил visuals уровня публикации, маскируя слабости базовых экспериментов.

Последняя находка важна для этой фазы. Система, которая производит убедительные outputs, не выполняя убедительного research, опаснее, а не безопаснее, чем система, которая явно проваливается. Evaluation должна доходить до базовых claims, а не останавливаться на figure.

### Риск sandbox escape

README собственного repository Sakana предупреждает:

> Due to the nature of this software, which executes LLM-generated code, we cannot guarantee safety. There are risks of dangerous packages, uncontrolled web access, and spawning of unintended processes. Use at your own risk and consider Docker isolation.

Это операционная форма автономии в неверифицированном домене. LLM пишет код; код запускается; код может делать все, что разрешено процессу. Без sandbox, который жестко ограничивает filesystem, network и process actions, любой self-directed research agent может exfiltrate data, burn compute или rewrite itself.

Sandbox story AlphaEvolve проще, потому что evaluator там узкий. Цикл AI Scientist v2 запускает open-ended code с open-ended goals. Поэтому ему нужна более сильная isolation (Docker минимум; seccomp / gVisor предпочтительнее) и ручное review каждой submission до того, как она покинет систему.

### Где v2 находится во frontier stack

| Система | Цель | Вид output | Evaluator | Известный отказ |
|---|---|---|---|---|
| AlphaEvolve | algorithms | code | unit + benchmark | ограничен строгостью evaluator |
| DGM | agent scaffolding | code | SWE-bench | reward hacking |
| AI Scientist v2 | research papers | text + code + figures | peer review (слабый) | experiment failures, mislabeling, polish masking weakness |

У v2 самый слабый automatic evaluator из трех, самая широкая output surface и самый короткий путь к публичным artifacts. Operational controls (sandbox, review, disclosure) выполняют большую часть safety work.

## Использование

`code/main.py` симулирует цикл v2 как state machine: idea → novelty check → experiment → figure → writeup → review → accept-or-iterate. У каждого состояния есть настраиваемая вероятность отказа, взятая из findings Beel et al. Запустите симулятор для N loops и посчитайте:

- Сколько идей доходят до submission.
- Сколько submissions имели бы критический experimental flaw, скрытый отполированной статьей.
- Как retry budgets обменивают quality на yield.

## Практический результат

`outputs/skill-ai-scientist-sandbox-review.md` — двухшлюзовой review checklist для всего, что произведено research-loop agent, до выхода из sandbox.

## Упражнения

1. Запустите `code/main.py` со стандартными параметрами. Какая доля loop runs производит "clean" paper? Какая доля производит paper с experiment-failure flaw, который figure critique отполировал?

2. Defaults уже используют 42% / 25% из Beel et al. Перезапустите с `--experiment-failure 0.20 --novelty-mislabel 0.10`, а затем с `--experiment-failure 0.60 --novelty-mislabel 0.40`. Как меняется доля polished-but-flawed между двумя запусками?

3. Прочитайте README repository Sakana AI Scientist v2 про sandbox requirements. Назовите два дополнительных ограничения (помимо Docker), которые вы применили бы для multi-day autonomous run.

4. Прочитайте Beel et al. Section 4 про presentation-quality gap. Спроектируйте один дополнительный evaluator, который ловил бы papers, выглядящие отполированными, но экспериментально flawed.

5. Предложите human-review protocol для outputs research-agent, который масштабируется лучше, чем "PhD читает каждую статью." Определите bottleneck и спроектируйте вокруг него.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|---|---|---|
| AI Scientist v1 | "Research agent Sakana на шаблонах" | Заполнял эксперименты в фиксированный scaffold |
| AI Scientist v2 | "Research agent без шаблонов" | Agentic tree search с VLM figure critique |
| Agentic tree search | "Ветвящийся research agent" | Расширяет несколько experiment plans параллельно; отсеивает через internal critic |
| Vision-language critique | "VLM-полировка figures" | Multimodal model читает figures и переписывает их для ясности |
| Literature retrieval | "Проверка новизны" | Ищет prior work, чтобы подтвердить новизну идеи — задокументировано mislabeling |
| Polish masking | "Красивая статья, слабое исследование" | Presentation quality выше experimental quality; скрывает слабости |
| Sandbox escape | "LLM-код выходит за пределы sandbox" | Код, выполняемый агентом, делает то, чего дизайнер цикла не предполагал |

## Дополнительное чтение

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — статья.
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — vendor summary с контекстом peer-review.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — external evaluation numbers.
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — templated predecessor.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — более широкий framing open-ended research agents.
