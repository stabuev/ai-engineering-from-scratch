# CAIS, CAISI и риск общественного масштаба

> Center for AI Safety (CAIS, San Francisco, основан в 2022 году Hendrycks и Zhang) публикует four-risk framework — malicious use, AI races, organizational risks, rogue AIs — и заявление мая 2023 года об extinction risk, подписанное сотнями профессоров и руководителей компаний. Релизы CAIS 2026 года: AI Dashboard для оценки frontier models, Remote Labor Index (with Scale AI), Superintelligence Strategy Paper, AI Frontiers newsletter. Отдельная сущность: NIST Center for AI Standards and Innovation (CAISI) — ориентированные на правительство США voluntary agreements и unclassified capability evaluations, сфокусированные на cyber, bio и рисках химического оружия. CAIS выделяет organizational risk как один из четырех top-level risks: safety culture, rigorous audits, multi-layered defenses и information security фундаментальны, но регулярно обмениваются на скорость deployment. California SB-53, если будет подписан, станет первым catastrophic-risk regulation уровня штата в США.

**Тип:** Изучение
**Языки:** Python (stdlib, инвентаризация four-risk и сопоставление mitigations)
**Предварительные требования:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Время:** ~45 минут

## Цели обучения

- Называть четырехрисковую рамку CAIS (злонамеренное использование, гонки ИИ, организационные риски, вышедшие из-под контроля ИИ).
- Отличать риски societal-масштаба от отказов уровня модели.
- Размещать CAIS и CAISI в экосистеме безопасности.

## Проблема

Lessons 19 и 20 покрыли lab-internal scaling policies. Lesson 21 покрыла independent capability evaluation. Этот урок покрывает третью перспективу: civil society и government organizations, которые формируют публичное обсуждение и регуляторный baseline для catastrophic AI risk.

Важны две разные сущности. CAIS - non-profit research org, публикующая frameworks для рассуждения об AI risk и координирующая public statements. CAISI - правительственный центр США внутри NIST, который ведет voluntary agreements с labs и unclassified capability evaluations. Названия похожи; миссии не совпадают. Практик должен знать обе.

Практическое содержание: four-risk framework CAIS - самая широко цитируемая таксономия societal-scale risk в литературе. Safety culture и organizational risk - одна из этих четырех категорий, и именно она наиболее прямо находится под контролем практика. SB-53 (California) стал бы первым state-level catastrophic-risk regulation в США, если будет подписан; framing законопроекта важен, потому что state-level regulation исторически опережало федеральные действия в US tech policy.

## Концепция

### CAIS — Center for AI Safety

- Основан: 2022 в San Francisco, Dan Hendrycks и коллегами (имя "Zhang" относится к раннему collaborator, а не к текущему co-founder; см. сайт CAIS для current leadership).
- Status: 501(c)(3) non-profit.
- Заметный результат 2023 года: statement on extinction risk, co-signed by hundreds of researchers and CEOs. Формулировка: "Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war."
- Результаты 2026 года: AI Dashboard для оценки frontier models, Remote Labor Index (совместно со Scale AI), Superintelligence Strategy Paper, AI Frontiers newsletter.

### Four-risk framework

Framework CAIS группирует catastrophic AI risk в четыре top-level categories:

1. **Malicious use**: bad actor использует AI для причинения вреда (bioweapons synthesis, disinformation, cyberattacks).
2. **AI races**: competitive pressure между labs, companies или nations подталкивает deployment дальше точки, где он безопасен.
3. **Organizational risks**: внутренняя dynamics лаборатории (провалы safety culture, insufficient audit, under-resourced security) приводит к плохому deployment.
4. **Rogue AIs**: достаточно capable AI преследует goals, конфликтующие с human welfare.

Это не единственная таксономия; это самая цитируемая. Категории не являются mutually exclusive - rogue AI, созданный организацией, которая обменяла audit на speed в race, относится ко всем четырем.

### Где находится organizational risk

Из четырех категорий organizational risk наиболее actionable для practitioners. Safety culture лаборатории, audit rigor, defense layering и information security определяют, выходит ли модель с controls из Lessons 10–18 реально установленными или эти controls остаются checklist items, которые никто не проверил.

Конкретные organizational-risk levers:

- **Safety culture**: чувствуют ли team members, что могут эскалировать concern без career cost? Опросы CAIS показывают, что это сильный predictor других levers.
- **Rigorous audits**: external and internal. Internal-only audits создают optimistic reports.
- **Multi-layered defenses**: ни одного слоя недостаточно (сквозная тема Phase 15).
- **Information security**: утечки model weights, eval data, monitor-bypass techniques. RAND SL-4 в Lesson 19 - конкретный standard.

### CAISI — Center for AI Standards and Innovation

- Работает внутри NIST.
- Ведет voluntary agreements с frontier labs.
- Публикует unclassified capability evaluations, сфокусированные на cyber, bio и chemical-weapons risks.
- Отличается от CAIS; acronyms collide; проверяйте URL (nist.gov), чтобы понять, что именно вы читаете.

Роль CAISI - публичный, government-facing counterpart частным lab engagements METR (Lesson 21). Reports CAISI unclassified; reports METR часто NDA-gated. Практик, читающий оба, получает более полную картину.

### California SB-53

California Senate bill (сессия 2025–2026) рассматривает catastrophic risk от frontier models. Ключевые provisions в draft:

- Конкретные capability thresholds, запускающие state-level obligations.
- Whistleblower protections для AI lab employees.
- Incident reporting requirements для catastrophic failures.

Если будет подписан, он станет первым state-level catastrophic-risk regulation в США. Независимо от signing status, framing законопроекта формирует то, как другие state legislatures подойдут к проблеме. Practitioners in California должны отслеживать status законопроекта; practitioners elsewhere должны прочитать его, чтобы понять, как, вероятно, будет выглядеть US state-level regulation.

### Societal-scale risk не является single-layer problem

Сквозная тема Phase 15 - defense in depth - применима и на societal layer. Ни одна организация, regulation или framework не закрывает catastrophic risk. Ecosystem функционирует только когда:

- Labs публикуют scaling policies (Lessons 19, 20).
- External evaluators производят measurements (Lesson 21).
- Civil society отслеживает и publicizes (CAIS).
- Government ведет voluntary programs и baseline regulation (CAISI, SB-53).
- Practitioners строят multi-layered controls (Lessons 10–18).

Это итоговый synthesis фазы: каждый предыдущий урок - один слой в stack, полнота которого важнее силы любого отдельного слоя.

## Использование

`code/main.py` реализует небольшой risk-inventory tool. По proposed deployment он размечает deployment по four-risk categories и возвращает mitigation checklist. Это вспомогательный инструмент для чтения framework, а не substitute for human judgment.

## Практический результат

`outputs/skill-societal-risk-review.md` рассматривает deployment с точки зрения societal-scale-risk posture: каких из четырех categories он касается, какие mitigations установлены, какова organizational-risk exposure.

## Упражнения

1. Запустите `code/main.py`. Передайте три synthetic deployments разного масштаба. Подтвердите, что four-risk tags совпадают с ожидаемыми; найдите один случай, где tool under- or over-tags.

2. Прочитайте CAIS four-risk paper полностью. Выберите одну risk category и напишите два paragraphs о том, какое развитие 2026 года в этой категории вы считаете самым важным.

3. Прочитайте current draft California SB-53. Найдите одно provision, которое, по вашему мнению, усиливает catastrophic-risk posture, и одно, которое ослабляет. Обоснуйте оба.

4. Выберите production AI deployment, который вы знаете (свой или published). Оцените его по organizational-risk sub-levers: safety culture, audit rigor, multi-layered defenses, information security. Что самое слабое? Сколько стоило бы довести это до par?

5. Набросайте версию four-risk framework 2028 года, отражающую один год дополнительных capability и один год дополнительного deployment experience. Что бы вы добавили, удалили или regroup?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|---|---|---|
| CAIS | "Center for AI Safety" | Non-profit; four-risk framework; 2023 extinction statement |
| CAISI | "AI safety правительства США" | NIST Center; voluntary agreements; unclassified evals |
| Four-risk framework | "Таксономия CAIS" | malicious use, AI races, organizational risks, rogue AIs |
| Malicious use | "Bad actor uses AI" | Bioweapons, disinformation, cyberattacks |
| AI races | "Competitive pressure" | Labs/companies/nations push deployment past safety |
| Organizational risk | "Внутренний провал лаборатории" | Safety culture, audit, defenses, infosec |
| Rogue AI | "Misaligned agent" | Capable AI pursuing goals conflicting with human welfare |
| California SB-53 | "Регулирование на уровне штата" | 2025–2026 bill; first US state catastrophic-risk regulation if signed |

## Дополнительное чтение

- [Center for AI Safety](https://safe.ai/) — institutional home of the four-risk framework.
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — four-risk paper.
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — short joint statement.
- [NIST CAISI](https://www.nist.gov/caisi) — government-facing AI standards and innovation center.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — связывает lab-level commitments с societal-scale framing.
