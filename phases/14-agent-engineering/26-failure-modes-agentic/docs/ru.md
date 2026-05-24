# Режимы отказа: почему агенты ломаются

> MASFT (Berkeley, 2025) каталогизирует 14 multi-agent failure modes в 3 категориях. Microsoft's Taxonomy документирует, как существующие AI failures усиливаются в agentic settings. Industry field data сходится на пяти повторяющихся modes: hallucinated actions, scope creep, cascading errors, context loss, tool misuse.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Время:** ~60 минут

## Цели обучения

- Назвать три категории failure в MASFT и минимум четыре конкретных modes в каждой.
- Объяснить, почему agentic failure усиливает существующие AI failure modes (bias, hallucination).
- Описать пять industry-recurring modes и их mitigations.
- Реализовать stdlib detector, который помечает agent traces метками failure-mode.

## Проблема

Команды выпускают agents, которые работают на 90% traces. Оставшиеся 10% failures - не случайный шум: они укладываются в небольшое число повторяющихся категорий. Когда вы можете их назвать, вы можете их мониторить и исправлять.

## Концепция

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy. 14 failure modes, сгруппированных в 3 категории. Inter-annotator Cohen's Kappa 0.88 — категории надежно различимы.

Центральный тезис: failures являются фундаментальными design flaws в multi-agent systems, а не ограничениями LLM, которые исправляются лучшими base models.

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- Существующие AI failures (bias, hallucination, data leakage) усиливаются в agentic settings.
- Новые failures возникают из autonomy: unintended action at scale, tool misuse, mission drift.
- Whitepaper - это risk register для agentic products.

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- Failures возникают из orchestration, internal state evolution и environment interaction.
- Это не просто "bad code" или "bad model output."

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

Два основных проявления:

1. **Instruction-following Deviation** — agent не следует system prompt.
2. **Long-range Contextual Misuse** — agent забывает или неверно применяет context из ранних turns.

Sub-intention errors: Omission (пропущенный step), Redundancy (повторенный step), Disorder (steps не по порядку).

### Пять industry-recurring modes

Field analyses Arize, Galileo, NimbleBrain за 2024-2026 сходятся на:

1. **Hallucinated actions.** Agent вызывает tool, которого не существует, или фабрикует arguments.
2. **Scope creep.** Agent расширяет задачу за пределы запроса пользователя (создает extra PRs, отправляет extra emails).
3. **Cascading errors.** Один неверный call запускает downstream effects. Галлюцинация phantom SKU запускает четыре API calls — multi-system incident.
4. **Context loss.** Long-horizon tasks забывают constraints из ранних turns.
5. **Tool misuse.** Вызывает правильный tool с неправильными arguments или вообще не тот tool.

Cascading - главный убийца. Agents не умеют отличать "I failed" от "the task is impossible" и часто hallucinate success message на 400 errors, чтобы закрыть loop.

### Mitigation: gates на каждом step

Automated verification gates на каждом step reasoning chain, проверяющие factual grounding против environment state. Конкретно:

- Per-step safety classifier (Lesson 21).
- Tool-call argument validation (Lesson 06).
- Cross-check retrieved content against known facts (Lesson 05, CRITIC).
- Detect success hallucination by re-probing state (was the file actually created?).

### Где failure monitoring ломается

- **Tagging only crashes.** Большинство agent failures производят valid-looking output. Нужны content-level checks.
- **Нет baseline.** Drift detection требует last-known-good; без него нельзя сказать "this is getting worse."
- **Over-alerting.** Каждый failure создает page. Кластеризуйте и rate-limit.

## Соберите это

`code/main.py` реализует stdlib failure-mode tagger:

- Synthetic trace dataset, покрывающий пять modes.
- Detector functions для каждого mode (signature patterns on tool calls, outputs, repeat actions).
- Tagger, который размечает каждый trace и сообщает mode distribution.

Запустите:

```
python3 code/main.py
```

Output: per-trace labels + aggregate distribution, дешевая репродукция того, что показывает Phoenix trace clustering.

## Используйте это

- **Phoenix** для production drift clustering (Lesson 24).
- **Langfuse** для session replay + annotation.
- **Custom** для domain-specific signatures, которые ваша observability platform не может обнаружить.

## Доведите до продакшена

`outputs/skill-failure-detector.md` генерирует failure-mode detectors, адаптированные к вашему domain и подключенные к trace store.

## Упражнения

1. Добавьте detector для "success hallucination": agent возвращает success, но target state не изменился.
2. Разметьте 100 real traces из продукта, который вы построили. Какой mode доминирует? Какова стоимость исправления?
3. Реализуйте metric "cascade radius": если failure произошел на step N, сколько downstream steps он затронул?
4. Прочитайте 14 failure modes MASFT. Выберите три, применимые к вашему продукту. Напишите detectors.
5. Подключите один detector к CI job: проваливайте build, если >=5% traces получают tag mode.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| MASFT | "Таксономия многоагентных отказов" | Берклийская классификация с 14 режимами |
| Cascading error | "Волновой отказ" | Одна ранняя ошибка распространяется через N шагов |
| Context loss | "Забыл ограничение" | Длинная задача теряет факты из ранних ходов |
| Tool misuse | "Не тот инструмент / не те аргументы" | Валидный вызов, но неверное применение |
| Success hallucination | "Фальшивое завершение" | Агент заявляет успех при 400; состояние не изменилось |
| Scope creep | "Выход за рамки" | Агент делает больше, чем просили |
| Instruction-following deviation | "Непослушание" | Игнорирует system prompt или пользовательское ограничение |
| Sub-intention errors | "Ошибки плана" | Пропуск, избыточность, нарушение порядка в исполнении плана |

## Дополнительное чтение

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure modes, 3 categories
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — реестр рисков
- [Arize Phoenix](https://docs.arize.com/phoenix) — drift clustering на практике
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — когда более простые patterns полностью избегают modes
