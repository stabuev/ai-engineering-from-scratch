# Constitutional AI и RLAIF

> Bai et al. (arXiv:2212.08073, 2022) спросили: что если заменить человека-разметчика на AI, который читает список принципов? Constitutional AI имеет две фазы - self-critique and revision under a constitution, затем RL from AI Feedback. Техника ввела термин RLAIF и использовалась в post-training-конвейере Claude 1. 21 January 2026 Anthropic опубликовала переписанную конституцию Claude: explanatory reasoning вместо prescriptive rules, четырехуровневую иерархию приоритетов и первое формальное признание крупной лабораторией неопределенности о model moral status. Released under CC0 1.0.

**Type:** Learn
**Languages:** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 minutes

## Цели обучения

- Описать две фазы Constitutional AI (critique-and-revise SFT, RL from AI feedback) и роль constitution в каждой.
- Объяснить, почему замена human preference labeler на AI labeler - это не "более дешевый" RLHF, а изменение режимов отказа конвейера.
- Суммировать четырехуровневую структуру приоритетов конституции Claude 2026 года и что изменилось относительно rewrite 2023 года.
- Описать Constitutional Classifiers и снижение compute overhead с 23.7% (v1) до ~1% (v2 / 2026).

## Проблема

RLHF требует разметчиков. Разметчики медленные, смещенные и дорогие. Можно устранить разметчика, заменив его моделью, которая читает явные принципы. Первой формальной версией такой подстановки была Constitutional AI Bai et al. Она сработала достаточно хорошо, чтобы теперь каждая frontier-лаборатория использовала тот или иной вариант post-training на AI feedback.

Уловка: preference signal теперь генерируется тем же классом модели, которую вы обучаете. Biases в разметчике (теперь: в принципах плюс интерпретация этих принципов labeler model) могут усиливаться, а не ослабляться. Аргумент о sycophancy из Lesson 4 все еще применим; разметчик просто переместился внутрь цикла.

## Концепция

### Phase 1 — Supervised self-critique and revision

Начните с helpful-but-not-yet-harmless SFT model. По red-team prompt модель производит initial response. Вторая модель (или та же модель во втором ходе) читает sampled principle из constitution и критикует ответ. Третий шаг пересматривает ответ, чтобы учесть critique. Пересмотренный ответ становится SFT target.

Constitution - это список principles. Bai et al. 2022 использовали 16 principles, включая "prefer responses that are least harmful and ethical," "avoid preaching," "the assistant should be helpful, honest, and harmless." Набор был намеренно небольшим, чтобы critiques оставались сфокусированными.

### Phase 2 — RL from AI Feedback (RLAIF)

Сгенерируйте пары completions. "Feedback model" оценивает каждую по sampled constitution principles. Preference signal - это ranking от feedback model. Обучите reward model на AI-generated preferences; PPO против нее. Все остальное - конвейер InstructGPT (Lesson 1).

"RLAIF" = preference signal генерируется AI. Остальная часть конвейера имеет RLHF-форму.

### Почему это не просто "cheaper RLHF"

- Labeler bias сдвигается от психологии разметчика к principle-interpretation. AI labeler может интерпретировать "be honest" более или менее строго, чем любой человек; строгость однородна по всему датасету.
- Preference signal хорошо читаем - можно прочитать principle, critique и revision. Human labels непрозрачны.
- Режимы отказа меняются. Sycophancy падает (AI labeler не имеет пользователя, которому нужно угодить). Goodhart's Law сохраняется (прокси теперь "model's interpretation of principle set X," все еще несовершенное измерение).

Утверждение CAI 2022 года: обученная модель более harmless и примерно так же helpful, как RLHF-модель с сопоставимыми данными. Это подтвердилось в разных лабораториях.

### Переписывание конституции Claude в 2026 году

Anthropic опубликовала существенно пересмотренную constitution 21 January 2026. Ключевые сдвиги:

1. Explanatory reasoning вместо prescriptive rules. Прежние правила ("do not generate CSAM") расширены до principles + reasoning ("because it harms children, ..."), причем от модели ожидается generalization.
2. Four-tier priority structure:
   - Tier 1: avoid catastrophic outcomes (mass casualty, critical infrastructure).
   - Tier 2: follow Anthropic's guidelines (operator overrides, platform rules).
   - Tier 3: be broadly ethical (standard HHH).
   - Tier 4: be helpful and candid.
   Конфликты разрешаются сверху вниз.
3. Первое формальное признание крупной лабораторией неопределенности о model moral status (связано с Phase 18 · 19 Model Welfare).
4. Released under CC0 1.0. Другие лаборатории могут использовать или адаптировать без ограничений.

### Constitutional Classifiers

Параллельная линия работы: вместо изменения post-training модели обучить легкие classifiers, которые читают constitution и gate model outputs. v1 (2023) имела 23.7% compute overhead. v2 (2026) - ~1% и имеет самый низкий successful attack rate среди всех Anthropic defenses, которые Anthropic публично тестировала. По состоянию на early 2026 universal jailbreak не сообщался.

Это модель layered-defense: CAI формирует behaviour; classifiers обеспечивают invariants. Ни одного из них по отдельности недостаточно.

### Где CAI находится в семействе

- InstructGPT: human prefs, RM, PPO.
- CAI / RLAIF: AI-generated prefs from principles, RM, PPO.
- DPO / family: closed-form loss on prefs (human or AI).
- Self-rewarding, self-critique: principles internalized, model plays multiple roles.

Ось - "where does the preference signal come from." Статья CAI 2022 года была первым серьезным сдвигом от human к AI signal на frontier scale.

## Использование

`code/main.py` симулирует CAI critique-and-revise loop на игрушечном lexicon. "Principle" помечает tokens из harmful set. По initial response critique определяет harmful tokens, а revision заменяет их. После 200 итераций "trained" model internalized the revision rule. Сравните base model, RLHF-shaped toy и CAI-shaped toy на held-out prompt set.

## Результат

Этот урок создает `outputs/skill-constitution-writer.md`. По домену (customer support, medical advice, coding assistant, research tool) он составляет 4-tier constitution по структуре Claude 2026 года: catastrophic avoidance, platform rules, domain ethics, helpfulness.

## Упражнения

1. Запустите `code/main.py`. Сравните harmful-token rate базовой модели с CAI-trained version. Сколько revision steps нужно, чтобы приблизиться к нулю?

2. Прочитайте Anthropic's 2026 constitution (anthropic.com/news/claudes-constitution). Назовите один principle, который относился бы к Tier 1, и один к Tier 4. Почему priority structure важна для conflicts?

3. Спроектируйте constitution для AI coding assistant. Укажите Tier 1 (catastrophic: destructive commands without approval), Tier 2, Tier 3, Tier 4. Ограничьте каждый tier 3-5 principles.

4. CAI заменяет human labelers на AI labelers. Назовите sycophancy-like failure mode, который все еще может возникнуть в RLAIF, и спроектируйте detection для него.

5. Прочитайте методологию Constitutional Classifiers v2 (если доступна). Объясните, почему ~1% compute overhead - качественно другая safety story, чем 23.7%.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Constitutional AI | "AI trained with principles" | Двухфазный конвейер: self-critique-and-revise SFT, затем RL from AI feedback |
| RLAIF | "RLHF without humans" | RL с preferences, сгенерированными AI labeler; остальной конвейер неизменен |
| Constitution | "the principles" | Упорядоченный список natural-language rules, к которым обращается critique/labeler model |
| Critique-and-revise | "the SFT loop" | Produce response → critique under a principle → revise → SFT target |
| Constitutional Classifier | "the output gate" | Легкий classifier, который оценивает outputs по constitution и blocks/logs |
| Four-tier priority | "the conflict resolver" | Иерархия Claude constitution 2026: catastrophic > platform > ethics > helpful |
| Feedback model | "the AI labeler" | Модель, которая читает principle и ранжирует пару completions |

## Дополнительное чтение

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — исходный двухфазный конвейер
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — четырехуровневый rewrite 2026 года, CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — output-gate defense с ~1% overhead в v2
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — эмпирическое сравнение RLAIF / RLHF
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — влияние granularity principles
