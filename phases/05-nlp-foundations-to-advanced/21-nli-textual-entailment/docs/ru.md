# Вывод на естественном языке (Natural Language Inference) — текстовое следование (Textual Entailment)

> "`t` entails `h`" означает, что человек, читающий `t`, заключил бы, что `h` истинно. NLI — задача предсказания entailment / contradiction / neutral. На поверхности скучно, но в продакшене это несущая конструкция.

**Тип:** Learn
**Языки:** Python
**Предварительные требования:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Время:** ~60 минут

## Цели обучения

- Запускать предобученную NLI-модель и применять ее для zero-shot классификации.
- Строить проверку фактологичности RAG через entailment.
- Различать entailment, contradiction и neutral.

## Проблема

Вы построили суммаризатор. Он выдал summary. Как узнать, что summary не содержит галлюцинации?

Вы построили чат-бота. Он ответил "yes." Как узнать, что ответ поддерживается извлеченным passage?

Вам нужно классифицировать 10 000 новостных статей по теме. У вас нет обучающих меток. Можно ли переиспользовать модель?

Все три проблемы сводятся к Natural Language Inference. NLI спрашивает: дана предпосылка `t` и гипотеза `h`; следует ли `h` из `t`, противоречит ей или является нейтральной (не связанной)?

- **Проверка галлюцинаций:** `t` = исходный документ, `h` = утверждение summary. Не entailment = галлюцинация.
- **Grounded QA:** `t` = извлеченный passage, `h` = сгенерированный ответ. Не entailment = fabrication.
- **Zero-shot classification:** `t` = документ, `h` = вербализованная метка ("This is about sports"). Entailment = предсказанная метка.

Одна задача, три продакшен-применения. Поэтому каждый фреймворк оценки RAG поставляется с NLI-моделью под капотом.

## Концепция

![NLI: трехклассовая классификация, premise vs hypothesis](../assets/nli.svg)

**Три метки.**

- **Entailment.** `t` → `h`. "The cat is on the mat" влечет "There is a cat."
- **Contradiction.** `t` → ¬`h`. "The cat is on the mat" противоречит "There is no cat."
- **Neutral.** Нет вывода ни в одну сторону. "The cat is on the mat" нейтрально к "The cat is hungry."

**Не логическое следование.** NLI — это *естественно-языковой* вывод (natural language inference): то, что вывел бы типичный читатель, а не строгая логика. "John walked his dog" в NLI влечет "John has a dog", но строгая логика первого порядка допустила бы это только если аксиоматизировать владение.

**Датасеты.**

- **SNLI** (2015). 570k пар с человеческой разметкой, подписи к изображениям как предпосылки. Узкий домен.
- **MultiNLI** (2017). 433k пар по 10 жанрам. Стандартный обучающий корпус в 2026 году.
- **ANLI** (2019). Adversarial NLI. Люди писали примеры, специально созданные для поломки существующих моделей. Сложнее.
- **DocNLI, ConTRoL** (2020–21). Предпосылки длиной в документ. Проверяет multi-hop и long-range inference.

**Архитектура.** Transformer encoder (BERT, RoBERTa, DeBERTa) читает `[CLS] premise [SEP] hypothesis [SEP]`. Представление `[CLS]` подается в 3-классовый softmax. Обучите на MNLI, оцените на held-out бенчмарках, получите 90%+ accuracy на in-distribution парах.

**Zero-shot через NLI.** Дан документ и кандидаты-метки; превратите каждую метку в гипотезу ("This text is about sports"). Вычислите вероятность entailment для каждой. Выберите максимум. Это механизм за pipeline `zero-shot-classification` в Hugging Face.

## Соберите это

### Шаг 1: запустите предобученную NLI-модель

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Для production NLI открытые варианты по умолчанию — `facebook/bart-large-mnli` и `microsoft/deberta-v3-large-mnli`. DeBERTa-v3 возглавляет leaderboard'ы.

### Шаг 2: zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Шаблон по умолчанию — "This example is about {label}." Настройте его через `hypothesis_template`. Обучающие данные не нужны. Fine-tuning не нужен. Работает из коробки.

### Шаг 3: проверка faithfulness для RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Это ядро faithfulness в RAGAS. Разбейте сгенерированный ответ на атомарные утверждения. Проверьте каждое утверждение относительно извлеченного контекста. Сообщите долю тех, которые следуют из контекста.

### Шаг 4: ручной NLI-классификатор (концептуально)

См. `code/main.py` для stdlib-only игрушки: premise и hypothesis сравниваются через lexical overlap + negation detection. Неконкурентно с transformer-моделями — но показывает форму задачи: два текста на входе, 3-классовая метка на выходе, loss = cross-entropy over `{entail, contradict, neutral}`.

## Ловушки

- **Hypothesis-only shortcuts.** Модели могут предсказать метку только по hypothesis примерно с 60% на SNLI, потому что "not", "nobody", "never" коррелируют с contradiction. Сильный baseline для обнаружения утечки меток.
- **Lexical overlap heuristic.** Эвристика подпоследовательности ("every subsequence is entailed") проходит SNLI, но проваливается на HANS/ANLI. Используйте adversarial benchmarks.
- **Деградация на document-length.** Single-sentence NLI-модели теряют 20+ F1 на предпосылках длиной в документ. Для длинного контекста используйте модели, обученные на DocNLI.
- **Чувствительность zero-shot template.** "This example is about {label}" vs "{label}" vs "The topic is {label}" может менять accuracy на 10+ пунктов. Настраивайте шаблон.
- **Domain mismatch.** MNLI обучается на общем английском. Юридические, медицинские и научные тексты требуют доменно-специфичных NLI-моделей (например, SciNLI, MedNLI).

## Используйте это

Стек 2026 года:

| Сценарий | Модель |
|---------|-------|
| NLI общего назначения | `microsoft/deberta-v3-large-mnli` |
| Быстро / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot классификация (легкая) | `facebook/bart-large-mnli` |
| NLI на уровне документа | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Многоязычность | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Обнаружение галлюцинаций в RAG | NLI-слой внутри RAGAS / DeepEval |

Мета-паттерн 2026 года: NLI — duct tape понимания текста. Когда вам нужно "does A support B?" или "does A contradict B?" — беритесь за NLI до того, как делать еще один LLM-вызов.

## Доведите до продакшена

Сохраните как `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Упражнения

1. **Легко.** Запустите `facebook/bart-large-mnli` на 20 вручную составленных triples (premise, hypothesis, label), покрывающих все три класса. Измерьте accuracy. Добавьте adversarial ловушки "subsequence heuristic" ("I did not eat the cake" vs "I ate the cake") и проверьте, сломается ли модель.
2. **Средне.** Сравните zero-shot template `"This text is about {label}"` с `"The topic is {label}"` и `"{label}"` на 100 заголовках AG News. Сообщите swing accuracy.
3. **Сложно.** Постройте RAG faithfulness checker: декомпозиция на atomic claims + NLI для каждого claim. Оцените на 50 RAG-generated answers с gold context. Измерьте false-positive и false-negative rates относительно hand labels.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-------------------|--------------------------------|
| NLI | Natural Language Inference | 3-классовая классификация отношения premise-hypothesis. |
| RTE | Recognizing Textual Entailment | Более старое название NLI; та же задача. |
| Entailment | "t implies h" | Типичный читатель заключил бы, что h истинно, если дано t. |
| Contradiction | "t rules out h" | Типичный читатель заключил бы, что h ложно, если дано t. |
| Neutral | "undecided" | Нет вывода от t к h ни в одну сторону. |
| Zero-shot classification | NLI как классификатор | Вербализовать метки как гипотезы, выбрать max entailment. |
| Faithfulness | Поддерживается ли ответ? | NLI по (retrieved context, generated answer). |

## Дополнительное чтение

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — benchmark ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — рабочая лошадка NLI в 2026 году.
