# Многоязычный NLP

> Одна модель, 100+ языков, ноль обучающих данных для большинства из них. Межъязыковой перенос (cross-lingual transfer) - практическое чудо 2020-х.

**Тип:** Обучение
**Языки:** Python
**Предварительные требования:** Фаза 5 · 04 (GloVe, FastText, Subword), Фаза 5 · 11 (Машинный перевод)
**Время:** ~45 минут

## Цели обучения

- Запускать zero-shot кросс-язычную классификацию многоязычной моделью.
- Объяснять общее многоязычное пространство эмбеддингов и стратегию few-shot дообучения.
- Объяснять налог токенизации для малоресурсных языков.

## Проблема

Для английского есть миллиарды размеченных примеров. Для урду - тысячи. Для майтхили почти нет. Любая практическая NLP-система, обслуживающая глобальную аудиторию, должна работать с длинным хвостом языков, где данных для обучения под конкретную задачу не существует.

Многоязычные модели решают это, обучая одну модель одновременно на многих языках. Общее представление позволяет модели переносить навыки, выученные на языках с большим объемом ресурсов, на низкоресурсные языки. Дообучите модель на анализе тональности на английском, и она сразу выдает удивительно хорошие предсказания тональности на урду. Это zero-shot межъязыковой перенос, и он изменил то, как NLP поставляется в мир.

Этот урок называет компромиссы, канонические модели и одно решение, на котором часто спотыкаются команды, впервые работающие с многоязычностью: выбор исходного языка для переноса.

## Концепция

![Межъязыковой перенос через общее многоязычное пространство эмбеддингов](../assets/multilingual.svg)

**Общий словарь.** Многоязычные модели используют токенизатор SentencePiece или WordPiece, обученный на текстах всех целевых языков. Словарь общий: одна и та же подсловная единица представляет одну и ту же морфему в родственных языках. `anti-` в английском и итальянском получает один и тот же токен.

**Общее представление.** Трансформер, предобученный с masked language modeling на многих языках, учится тому, что семантически похожие предложения на разных языках дают похожие скрытые состояния. mBERT, XLM-R и NLLB все демонстрируют это. Эмбеддинги для "cat" в английском группируются рядом с "chat" во французском и "gato" в испанском; то же происходит и с эмбеддингами целых предложений.

**Zero-shot перенос.** Дообучите модель на размеченных данных одного языка (обычно английского). При инференсе запустите ее на любом другом языке, который поддерживает модель. Разметка на целевом языке не нужна. Результаты сильны для типологически родственных языков и слабее для далеких.

**Few-shot дообучение.** Добавьте 100-500 размеченных примеров на целевом языке. На задачах классификации точность подскакивает до 95-98% от английского бейзлайна. Это самый экономически эффективный рычаг в многоязычном NLP.

## Модели

| Модель | Год | Покрытие | Примечания |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | Обучен на Wikipedia. Первая практичная многоязычная LM. Слаб на низкоресурсных языках. |
| XLM-R | 2019 | 100 languages | Обучен на CommonCrawl (намного больше Wikipedia). Задает межъязыковой бейзлайн. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 languages | XLM-R со словарем в 1M токенов (вместо 250k). Лучше на низкоресурсных языках. |
| mT5 | 2020 | 101 languages | Архитектура T5 для многоязычной генерации. |
| NLLB-200 | 2022 | 200 languages | Модель перевода Meta; включает 55 низкоресурсных языков. |
| BLOOM | 2022 | 46 languages + 13 programming | Открытая 176B LLM, обученная многоязычно. |
| Aya-23 | 2024 | 23 languages | Многоязычная LLM от Cohere. Сильна на арабском, хинди, суахили. |

Выбирайте по случаю использования. Классификация хорошо работает с XLM-R-base как разумным вариантом по умолчанию. Для задач генерации нужны mT5 или NLLB в зависимости от того, это перевод или открытая генерация. Работа в стиле LLM сочетается с Aya-23 или Claude при явном многоязычном prompting.

## Решение об исходном языке (исследования 2026)

Большинство команд по умолчанию используют английский как исходный язык для дообучения. Недавние исследования (2026) показывают, что это часто неверно.

Языковое сходство предсказывает качество переноса лучше, чем сырой размер корпуса. Для славянских целевых языков немецкий или русский часто превосходят английский. Для индийских целевых языков хинди часто превосходит английский. Метрика сходства **qWALS** (2026, основана на признаках World Atlas of Language Structures) количественно оценивает это. **LANGRANK** (Lin et al., ACL 2019) - отдельный, более ранний метод, который ранжирует кандидаты в исходные языки по сочетанию лингвистического сходства, размера корпуса и генетического родства.

Практическое правило: если у вашего целевого языка есть типологически близкий высокоресурсный родственник, сначала попробуйте дообучение на нем, затем сравните с дообучением на английском.

## Соберите это

### Шаг 1: zero-shot межъязыковая классификация

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Одна модель, три языка, тот же API. XLM-R, обученная на данных NLI, хорошо переносится на классификацию через прием с entailment.

### Шаг 2: многоязычное пространство эмбеддингов

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Переводы оказываются близко в пространстве эмбеддингов. Другое английское предложение оказывается дальше. Именно это заставляет работать межъязыковой поиск, кластеризацию и оценку сходства.

### Шаг 3: стратегия few-shot дообучения

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Для 100-500 примеров на целевом языке `num_train_epochs=5` и `learning_rate=2e-5` - безопасные значения по умолчанию. Более высокие learning rates приводят к разрушению многоязычного выравнивания, и вы получаете модель только для английского.

## Оценивание, которое действительно работает

- **Точность по каждому языку на отложенных наборах.** Не агрегированная. Агрегат скрывает длинный хвост.
- **Сравнение с одноязычным бейзлайном.** Для языков с достаточным объемом данных одноязычная модель, обученная с нуля, иногда превосходит многоязычную. Проверяйте.
- **Тесты на уровне сущностей.** Именованные сущности на целевом языке. У многоязычных моделей часто слабая токенизация для письменностей, далеких от латиницы.
- **Межъязыковая согласованность.** Один и тот же смысл на двух языках должен давать одно и то же предсказание. Измеряйте разрыв.

## Используйте это

Стек 2026:

| Задача | Рекомендуется |
|-----|-------------|
| Классификация, 100 языков | Дообученная XLM-R-base (~270M) |
| Zero-shot классификация текста | `joeddav/xlm-roberta-large-xnli` |
| Многоязычные sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Перевод, 200 языков | `facebook/nllb-200-distilled-600M` (см. урок 11) |
| Генеративная многоязычность | Claude, GPT-4, Aya-23, mT5-XXL |
| NLP для низкоресурсного языка | XLM-V или предметно-специфичное дообучение на родственном высокоресурсном языке |

Всегда закладывайте бюджет на дообучение на целевом языке, если качество важно. Zero-shot - это отправная точка, а не окончательный ответ.

### Налог токенизации (что ломается для низкоресурсных языков)

Многоязычные модели используют один токенизатор для всех своих языков. Этот словарь обучается на корпусе, где доминируют английский, французский, испанский, китайский, немецкий. Для любого языка вне доминирующего набора три налога незаметно складываются:

- **Налог фертильности (fertility tax).** Текст низкоресурсного языка токенизируется в гораздо больше токенов на слово, чем английский. Предложению на хинди может понадобиться в 3-5x больше токенов, чем эквивалентному английскому предложению. Эти 3-5x съедают ваше контекстное окно, эффективность обучения и задержку.
- **Налог восстановления вариантов.** Каждая опечатка, вариант диакритики, несовпадение Unicode-нормализации или изменение регистра становится холодным стартом как неродственная последовательность в пространстве эмбеддингов. Модель не может выучить орфографические соответствия, которые носитель языка воспринимает как очевидные.
- **Налог перерасхода емкости.** Налоги 1 и 2 расходуют позиции контекста, глубину слоев и размерности эмбеддингов. То, что остается для собственно рассуждения, систематически меньше того, что высокоресурсный язык получает от той же модели.

Практический симптом: ваша модель нормально обучается на хинди, кривая потерь выглядит правильно, eval perplexity выглядит разумно, а production-выводы тонко ошибочны. Морфология разваливается в середине предложения. Редкие словоизменительные формы остаются невосстановимыми. **Вы не сможете выбраться из сломанного токенизатора простым масштабированием данных.**

Смягчения: выберите токенизатор с хорошим покрытием для целевого языка (словарь XLM-V на 1M токенов - прямое исправление); проверьте фертильность токенизации на отложенном целевом тексте до обучения; используйте byte-level fallback (SentencePiece `byte_fallback=True`, GPT-2-style byte-level BPE) для действительно длиннохвостых письменностей, чтобы ничего никогда не становилось OOV.

## Доведите до поставки

Сохраните как `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Упражнения

1. **Легко.** Запустите zero-shot classification pipeline на 10 предложениях на язык для английского, французского, хинди и арабского. Сообщите точность по каждому. Вы должны увидеть сильный французский, приличный хинди, переменный арабский.
2. **Средне.** Используйте `paraphrase-multilingual-MiniLM-L12-v2`, чтобы построить межъязыковой retriever по небольшому смешанному многоязычному корпусу. Делайте запрос на английском, извлекайте документы на любом языке. Измерьте recall@5.
3. **Сложно.** Сравните дообучение с английским источником и хинди-источником для задачи классификации на хинди. Используйте 500 примеров на целевом языке для few-shot дообучения в обоих режимах. Сообщите, какой источник дает лучшую точность на хинди и на сколько. Это тезис LANGRANK в миниатюре.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| Multilingual model | Одна модель, много языков | Общий словарь и параметры между языками. |
| Cross-lingual transfer | Обучите на одном языке, запустите на другом | Дообучение на источнике, оценка на цели без разметки целевого языка. |
| Zero-shot | Нет разметки целевого языка | Перенос без дообучения на целевом языке. |
| Few-shot | Малый объем целевой разметки | 100-500 примеров на целевом языке, используемых для дообучения. |
| mBERT | Первая многоязычная LM | BERT на 104 языках, предобученный на Wikipedia. |
| XLM-R | Стандартный межъязыковой бейзлайн | RoBERTa на 100 языках, предобученная на CommonCrawl. |
| NLLB | 200-язычная MT от Meta | No Language Left Behind. Включает 55 низкоресурсных языков. |

## Дополнительное чтение

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — статья XLM-R.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — аналитическая статья, с которой началась исследовательская линия межъязыкового переноса.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — статья NLLB-200.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya, многоязычная LLM от Cohere.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — статья qWALS / LANGRANK об исходном языке.
