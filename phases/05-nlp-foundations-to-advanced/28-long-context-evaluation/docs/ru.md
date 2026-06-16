# Оценка длинного контекста — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro заявляет 10M токенов контекста. При 1M токенов 8-needle MRCR падает до 26.3%. Заявленное ≠ пригодное. Оценка длинного контекста показывает фактическую емкость модели, которую вы выводите в продукт.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 5 · 13 (Question Answering), Фаза 5 · 23 (Chunking Strategies)
**Время:** ~60 минут

## Цели обучения

- Строить доменный Needle-in-a-Haystack и multi-needle вариант.
- Запускать multi-hop трассировку переменных (в стиле RULER) и LongBench на своём стеке.
- Отличать рекламируемый контекст от эффективного (lost-in-the-middle, MRCR).

## Проблема

У вас есть контракт на 200 страниц. Модель заявляет контекст в 1M токенов. Вы вставляете контракт и спрашиваете: "What is the termination clause?" Модель отвечает, но берет ответ с титульной страницы, потому что пункт о расторжении находится на глубине 120k токенов, дальше той области, куда модель фактически обращает внимание.

Это разрыв емкости контекста в 2026 году. Спецификации говорят 1M или 10M. Реальность говорит, что пригодны 60-70% от этого, а "пригодность" зависит от задачи.

- **Извлечение (одна игла в стоге сена):** почти идеально до заявленного максимума на frontier-моделях.
- **Многошаговые выводы / агрегация:** резко деградирует после ~128k на большинстве моделей.
- **Рассуждение по разнесенным фактам:** первая задача, которая ломается.

Оценка длинного контекста измеряет эти оси. Этот урок называет бенчмарки, объясняет, что именно измеряет каждый из них, и показывает, как построить кастомный needle-тест для вашей предметной области.

## Концепция

![Базовый NIAH, многозадачный RULER, целостный LongBench](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Поместите факт ("the magic word is pineapple") на контролируемую глубину в длинный контекст. Попросите модель извлечь его. Пройдитесь по сетке depth × length. Исходный бенчмарк длинного контекста. Frontier-модели теперь насыщают его; это необходимая, но недостаточная базовая проверка.

**RULER (Nvidia, 2024).** 13 типов задач в 4 категориях: извлечение (single / multi-key / multi-value), многошаговое трассирование (отслеживание переменных), агрегация (частота общих слов), QA. Настраиваемая длина контекста (от 4k до 128k+). Выявляет модели, которые насыщают NIAH, но проваливаются на многошаговых задачах. В релизе 2024 года только половина из 17 моделей, заявлявших контекст 32k+, сохраняла качество на 32k.

**LongBench v2 (2024).** 503 вопроса с множественным выбором, контексты длиной 8k-2M слов, шесть категорий задач: single-doc QA, multi-doc QA, long in-context learning, long dialogue, code repo, long structured data. Производственный бенчмарк для поведения длинного контекста в реальном мире.

**MRCR (Multi-Round Coreference Resolution).** Многоходовая кореференция в масштабе. Варианты 8-needle, 24-needle, 100-needle. Показывает, сколькими фактами модель может оперировать до деградации внимания.

**NoLiMa.** "Non-lexical needle." Игла и запрос не имеют буквального пересечения; извлечение требует одного шага семантического рассуждения. Сложнее, чем NIAH.

**HELMET.** Конкатенирует много документов и задает вопрос по любому одному из них. Тестирует избирательное внимание.

**BABILong.** Встраивает цепочки рассуждений bAbI в нерелевантные стоги сена. Тестирует reasoning-in-a-haystack, а не только извлечение.

### Что действительно нужно отчитывать

- **Заявленное окно контекста.** Число из спецификации.
- **Эффективная длина извлечения.** Прохождение NIAH при заданном пороге (например, 90%).
- **Эффективная длина рассуждения.** Прохождение многошаговой задачи или агрегации при том же пороге.
- **Кривая деградации.** Accuracy vs context length, построенная по типам задач.

Два числа для вашей спецификации: retrieval-effective и reasoning-effective. Обычно reasoning-effective составляет 25-50% от заявленного окна.

## Соберите это

### Шаг 1: кастомный NIAH для вашей предметной области

См. `code/main.py`. Каркас:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Пройдитесь по `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Постройте тепловую карту. Это NIAH-карточка для вашей целевой модели.

### Шаг 2: вариант с несколькими иглами

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Вопросы вроде "What are the three magic words?" требуют извлечь все три. Успех с одной иглой не предсказывает успех с несколькими иглами.

### Шаг 3: многошаговое трассирование переменных (в стиле RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Ответ требует связать цепочку из трех присваиваний. Frontier-модели на 128k здесь часто падают до 50-70% accuracy.

### Шаг 4: LongBench v2 на вашем стеке

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Отчитывайте accuracy по категориям. Агрегированные оценки скрывают большие различия на уровне задач.

## Подводные камни

- **Оценка только по NIAH.** Прохождение NIAH на 1M токенов ничего не говорит о многошаговых задачах. Всегда запускайте RULER или кастомный многошаговый тест.
- **Равномерная выборка глубины.** Многие реализации проверяют только depth=0.5. Проверяйте depth=0, 0.25, 0.5, 0.75, 1.0 — эффект "lost in the middle" реален.
- **Лексическое пересечение с filler.** Если игла имеет общие ключевые слова с filler, извлечение становится тривиальным. Используйте NoLiMa-style иглы без пересечений.
- **Игнорирование задержки.** Промпты на 1M токенов требуют 30-120 секунд на prefill. Измеряйте time-to-first-token вместе с accuracy.
- **Числа, самостоятельно заявленные вендором.** OpenAI, Google, Anthropic публикуют собственные оценки. Всегда независимо перезапускайте оценку на своем use case.

## Используйте это

Стек 2026 года:

| Ситуация | Бенчмарк |
|-----------|-----------|
| Быстрая sanity-проверка | Кастомный NIAH на 3 глубинах × 3 длинах |
| Выбор модели для продакшена | RULER (13 задач) на вашей целевой длине |
| Качество QA в реальном мире | Подмножество LongBench v2 single-doc-QA |
| Многошаговое рассуждение | BABILong или кастомное трассирование переменных |
| Разговоры / диалог | MRCR 8-needle на вашей целевой длине |
| Регрессия при обновлении модели | Фиксированный внутренний harness NIAH + RULER, запуск на каждой новой модели |

Правило для продакшена: никогда не доверяйте окну контекста, пока не проверили NIAH + 1 задачу на рассуждение на вашей предполагаемой длине.

## Отгрузите это

Сохраните как `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Упражнения

1. **Легко.** Постройте NIAH с 3 глубинами (0.25, 0.5, 0.75) × 3 длинами (1k, 4k, 16k). Запустите на любой модели. Постройте pass rate в виде тепловой карты 3×3.
2. **Средне.** Добавьте вариант с 3 иглами. Измерьте извлечение всех 3 на каждой длине. Сравните с pass rate для одной иглы на той же длине.
3. **Сложно.** Сконструируйте задачу трассирования переменных (X1 → X2 → X3, с 3 шагами), встроенную в 64k filler. Измерьте accuracy на 3 frontier-моделях. Отчитайте effective reasoning length для каждой модели.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| NIAH | Needle in haystack | Поместить факт в filler и попросить модель извлечь его. |
| RULER | NIAH на стероидах | 13 типов задач по извлечению / multi-hop / агрегации / QA. |
| Effective context | Реальная емкость | Длина, на которой accuracy все еще держится выше порога. |
| Lost in the middle | Смещение по глубине | Модели уделяют недостаточно внимания содержимому в середине длинных входов. |
| Multi-needle | Много фактов сразу | Несколько вставок; тестирует удержание внимания между фактами, а не только извлечение. |
| MRCR | Multi-round coref | Кореференция с 8, 24 или 100 иглами; выявляет насыщение внимания. |
| NoLiMa | Non-lexical needle | Игла и запрос не имеют буквальных общих токенов; требуется рассуждение. |

## Дополнительное чтение

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — исходный репозиторий NIAH.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — многозадачный бенчмарк.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — оценка длинного контекста в реальном мире.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — более сложные иглы.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — статья о смещении по глубине.
