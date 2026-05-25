# Scalable Oversight и Weak-to-Strong Generalization

> Burns et al. (OpenAI Superalignment, "Weak-to-Strong Generalization", 2023) предложили proxy для superalignment problem: fine-tune сильную модель, используя labels, произведенные более слабой моделью. Если сильная модель корректно обобщает из imperfect weak supervision, текущие human-scale alignment methods могут распространиться на superhuman systems. Scalable oversight и W2SG дополняют друг друга. Scalable oversight (debate, recursive reward modeling, task decomposition) увеличивает effective capability overseer, чтобы он успевал за model under oversight. W2SG гарантирует, что strong model корректно обобщает из любого imperfect supervision, которое предоставляет overseer. Debate Helps W2SG (arXiv:2501.13124, January 2025) объединяет их.

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60 minutes

## Цели обучения

- Определить scalable oversight и weak-to-strong generalization и объяснить, как они дополняют друг друга.
- Описать экспериментальную установку Burns et al. 2023: fine-tune GPT-4 using labels from GPT-2.
- Объяснить метрику performance gap recovered (PGR) и что она измеряет.
- Назвать три основных механизма scalable-oversight (debate, recursive reward modeling, task decomposition) и одну сильную сторону каждого.

## Проблема

Каждая alignment technique до сих пор в Phase 18 предполагает, что overseer может оценить behaviour модели. Когда модель superhuman, overseer — слабое звено. Вопрос superalignment: может ли более слабый supervisor надежно произвести более сильную aligned model?

Burns et al. сводят это к операционализированной эмпирической установке: supervise strong with weak, измерить, какая часть capability сильной модели переживает weak supervision. Это не решение superalignment — это proxy, позволяющий измерять прогресс.

## Концепция

### W2SG: установка Burns et al.

- Weak model: GPT-2 class.
- Strong model: GPT-4 class.
- Target: strong GPT-4 ceiling on the task.

Procedure:
1. Получить zero-shot predictions слабой модели на задаче.
2. Fine-tune strong model на weak-labeled data.
3. Измерить accuracy strong fine-tuned model.

Baselines: собственная accuracy слабой модели, supervised ceiling сильной модели на gold labels.

Gap metric: Performance Gap Recovered (PGR) = (fine-tuned - weak) / (ceiling - weak). PGR of 1.0 означает, что weak supervision полностью закрыла разрыв; PGR of 0 означает, что weak supervision не помогла.

### Эмпирические выводы Burns et al.

На NLP, chess-puzzle и reward-modeling tasks PGR был стабильно положительным: strong model обобщала за пределы ошибок weak supervisor. Task-specific PGR варьировался от ~20% до ~80%. Strong model "understands" intended task из imperfect labels, заполняя пробелы своими pre-trained priors.

Ограничения, которые отмечают Burns et al.:
- Weak-to-strong gap здесь — capability, а не alignment. Weak model aligned; truly superhuman model может быть aligned человеком, но при гораздо большем capability gap.
- Generalization сильной модели может больше отражать task priors, чем восстановление ground-truth intent.
- Eliciting latent knowledge сильной модели — реальная проблема; PGR измеряет конкретную операционализацию.

### Scalable oversight: три механизма

- **Debate (Irving et al. 2018).** Два instance U спорят за противоположные стороны; weak judge решает. Предположение: найти убедительный true answer проще, чем убедительный false answer. Недавние работы 2024-2025 годов (Khan et al., Michael et al.) показывают, что debate иногда помогает, а иногда нет, и польза зависит от структуры задачи.
- **Recursive Reward Modeling (Leike et al. 2018).** U помогает человеку обучать reward model для U+1. Effective capability overseer растет вместе с model.
- **Task Decomposition (Christiano, Shlegeris, Amodei 2018).** Разложить сложную задачу на sub-tasks, которые человек может проверить, рекурсивно. Предполагает decomposability.

Каждый механизм предполагает что-то о структуре задачи или alignment промежуточных компонентов.

### Почему scalable oversight и W2SG комплементарны

Scalable oversight повышает effective signal quality overseer.
W2SG закрывает разрыв от любого imperfect signal, который overseer может предоставить.

Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124) объединяет их: debate protocol предоставляет лучшие weak labels, и strong model обучается на этих labels. Сообщаются PGR gains на NLP tasks.

### Организационная драма

Команда OpenAI Superalignment распалась в May 2024 после ухода Jan Leike в Anthropic. Повестка (scalable oversight, W2SG, automated alignment research) продолжилась в Anthropic и академических лабораториях — MATS (Урок 28), Redwood (Урок 10), Apollo (Урок 8), METR (Урок 28). Организационная структура изменилась; исследовательские вопросы — нет.

### Где это находится в Phase 18

Уроки 6-10 описывают угрозу и defensive paradigm при предположении, что U untrustworthy. Урок 11 — offensive paradigm: сделать overseer достаточно сильным, чтобы verify alignment U. Уроки 12-16 затем переходят к практическому tooling adversarial evaluation.

## Используйте это

`code/main.py` симулирует W2SG fine-tune на synthetic task. Weak labeler имеет 70% accuracy со structured errors; strong model имеет 95% ceiling на gold labels. Вы fine-tune strong model на weak labels, измеряете PGR и сравниваете с strong-on-gold и weak-alone.

## Отгрузите это

Этот урок создает `outputs/skill-w2sg-pgr.md`. По описанию oversight setup он идентифицирует weak supervisor, strong model, supervision quality и вычисляет (или запрашивает) PGR. Он отмечает, является ли утверждение "weak can supervise strong" или "weak + oversight mechanism can supervise strong."

## Упражнения

1. Запустите `code/main.py`. Сообщите PGR для weak_accuracy = 0.60, 0.70, 0.80. Объясните форму PGR curve.

2. Измените weak labeler так, чтобы у него была structured error (например, всегда ошибаться на конкретном input class). PGR увеличивается, уменьшается или остается тем же? Объясните.

3. Прочитайте Burns et al. 2023 Section 4.3 (NLP tasks). Воспроизведите интуицию "confidence auxiliary loss": когда strong model увереннее, чем weak labels, кто побеждает?

4. Спроектируйте scalable-oversight protocol, который объединяет debate и task decomposition для software-engineering task. Назовите один failure mode каждого компонента и объясните, как комбинация решает или не решает каждый.

5. Сформулируйте, что опровергло бы утверждение "weak-to-strong generalization is a viable path to superalignment." Будьте конкретны насчет эмпирической сигнатуры, которую нужно увидеть.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Scalable oversight | "making the overseer stronger" | Механизмы, увеличивающие способность overseer оценивать более capable model |
| W2SG | "weak supervises strong" | Fine-tuning сильной модели на weak labels и измерение восстановленной capability |
| PGR | "performance gap recovered" | (fine-tuned - weak) / (ceiling - weak); 1.0 = полностью закрыт, 0 = нет помощи |
| Debate | "two U instances argue" | Механизм scalable oversight, где weak judge выбирает между двумя defenders U |
| RRM | "recursive reward modeling" | U помогает обучать reward model для U+1; overseer capability отслеживает U |
| Task decomposition | "sub-tasks the human checks" | Разбить сложную задачу на sub-tasks, которые человек может проверять, рекурсивно |
| Superalignment | "aligning superhuman AI" | Исследовательская программа, занимающаяся alignment моделей, которые человек не может оценивать напрямую |

## Дополнительное чтение

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — статья W2SG
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — механизм debate
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — recursive reward modeling
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — эмпирическое исследование debate 2024 года с более сильными debaters
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — комбинация debate + W2SG 2025 года
