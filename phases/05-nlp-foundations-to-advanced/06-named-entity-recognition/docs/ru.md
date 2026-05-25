# Распознавание именованных сущностей

> Извлечь имена. Звучит просто, пока не сталкиваешься с неоднозначными границами, вложенными сущностями и доменной терминологией.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Время:** ~75 минут

## Проблема

"Apple sued Google over its iPhone search deal in the US." Пять сущностей: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (возможно), US (GPE). Хорошая NER-система извлекает их все с корректными типами. Плохая пропускает iPhone, путает Apple-фрукт с Apple-компанией и размечает "US" как PERSON.

NER - рабочая лошадка под каждым пайплайном структурированного извлечения. Парсинг резюме, сканирование compliance-логов, анонимизация медицинских записей, понимание поисковых запросов, grounding для ответов чатботов, извлечение данных из юридических договоров. Вы почти никогда не видите NER напрямую, но постоянно от него зависите.

Этот урок проходит классический путь (rule-based, HMM, CRF) к современному (BiLSTM-CRF, затем transformers). Каждый шаг решает конкретное ограничение предыдущего. Этот шаблон и есть главный урок.

## Концепция

![NER-разметка: схема BIO + конвейер CRF+BiLSTM](./assets/ner.svg)

**BIO tagging** (или BILOU) превращает извлечение сущностей в задачу sequence labeling. Каждому токену назначается метка `B-TYPE` (начало сущности), `I-TYPE` (внутри сущности) или `O` (вне любой сущности).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Многотокенные сущности образуют цепочку: `New B-GPE`, `York I-GPE`, `City I-GPE`. Модель, понимающая BIO, может извлекать произвольные spans.

Развитие архитектур:

- **Rule-based.** Regex + gazetteer lookups. Высокая precision на известных сущностях, нулевое покрытие новых.
- **HMM.** Hidden Markov Model. Вероятность эмиссии токена при заданной метке, вероятность перехода от метки к метке. Декодирование Viterbi. Обучается на размеченных данных.
- **CRF.** Conditional Random Field. Похожа на HMM, но дискриминативная, поэтому можно смешивать произвольные признаки (форма слова, капитализация, соседние слова). В 2026 году все еще классическая production-рабочая лошадка для low-resource deployment.
- **BiLSTM-CRF.** Нейронные признаки вместо hand-crafted. LSTM читает предложение в обоих направлениях, CRF-слой сверху обеспечивает согласованные последовательности меток.
- **Transformer-based.** Fine-tune BERT с token-classification head. Лучшая точность. Больше всего compute.

## Сборка

### Шаг 1: помощники для BIO tagging

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Шаг 2: hand-crafted features

Для классического (не нейронного) NER признаки решают все. Полезные признаки:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` возвращает `xXxxxx`. `word_shape("USA-2024")` возвращает `XXX-dddd`. Паттерны капитализации дают сильный сигнал для имен собственных.

### Шаг 3: простой rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Production-gazetteers содержат миллионы записей, собранных из Wikipedia и DBpedia. Покрытие хорошее. Disambiguation (`Apple` компания или фрукт) ужасен. Поэтому статистические модели победили.

### Шаг 4: шаг к CRF (набросок, не полная реализация)

Полный CRF с нуля в 50 строк не дает понимания без оснований теории вероятностей. Вместо этого используйте `sklearn-crfsuite`:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` и `c2` - L1 и L2 regularization. `all_possible_transitions=True` позволяет модели выучить, что незаконные последовательности (например, `I-ORG` после `O`) маловероятны. Так CRF обеспечивает BIO-consistency без ручного кодирования ограничения.

### Шаг 5: что добавляет BiLSTM-CRF

Признаки становятся learned. Входы: token embeddings (GloVe или fastText). LSTM читает предложение слева направо и справа налево. Конкатенированные hidden states проходят через CRF output layer. CRF по-прежнему обеспечивает согласованность последовательности меток; LSTM заменяет hand-crafted features на learned features.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Для CRF layer используйте `torchcrf.CRF` (pip install pytorch-crf). Прирост над hand-crafted CRF измерим, но меньше, чем можно ожидать, если у вас нет десятков тысяч размеченных предложений.

## Использование

spaCy поставляет production-grade NER из коробки.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Обратите внимание: `iPhone` размечен как `ORG`, а не как `PRODUCT` - у малой модели spaCy слабое покрытие product-entities. Большая модель (`en_core_web_lg`) справляется лучше. Transformer model (`en_core_web_trf`) еще лучше.

Hugging Face для BERT-based NER:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` объединяет соседние B-X, I-X tokens в span. Без него вы получите token-level labels и будете объединять их сами.

### LLM-based NER (вариант 2026 года)

Zero-shot и few-shot LLM NER теперь конкурентоспособны с fine-tuned models во многих доменах и значительно лучше, когда размеченных данных мало.

- **Zero-shot prompting.** Дайте LLM список типов сущностей и пример схемы. Попросите JSON output. Работает из коробки; точность на новых доменах средняя.
- **ZeroTuneBio-style prompting.** Разложите задачу на candidate extraction -> meaning explanation -> judgment -> re-check. Многоэтапный prompt (не one-shot) заметно повышает точность на biomedical NER. Тот же паттерн работает для юридических, финансовых и научных доменов.
- **Dynamic prompting with RAG.** Для каждого inference call извлекайте самые похожие размеченные примеры из небольшого annotated seed set; собирайте few-shot prompt на лету. В бенчмарках 2026 года это повышает biomedical NER F1 для GPT-4 на 11-12% по сравнению со static prompting.
- **Per-entity-type decomposition.** Для длинных документов один вызов, извлекающий все типы сущностей сразу, теряет recall с ростом длины. Запускайте отдельный extraction pass для каждого типа сущности. Стоимость inference выше, точность существенно выше. Это стандартный паттерн для clinical notes и legal contracts.

Production-рекомендация на 2026 год: начните с LLM zero-shot baseline до сбора training data. Часто F1 уже достаточно хорош, и fine-tune не понадобится.

### Где классический NER все еще выигрывает

Даже при наличии LLM классический NER выигрывает, когда:

- Latency budget меньше 50ms.
- У вас есть тысячи размеченных примеров и нужен 98%+ F1.
- В домене стабильная онтология, где pretrained CRF или BiLSTM хорошо переносится.
- Regulatory constraints требуют on-prem, non-generative model.

### Где все разваливается

- **Domain shift.** NER, обученный на CoNLL, на юридических договорах работает хуже gazetteer. Fine-tune на вашем домене.
- **Nested entities.** "Bank of America Tower" одновременно ORG и FACILITY. Стандартный BIO не может представить overlapping spans. Нужен nested NER (multi-pass или span-based models).
- **Long entities.** "United States Federal Deposit Insurance Corporation." Token-level models иногда разбивают это. Используйте `aggregation_strategy` или post-process.
- **Sparse types.** Медицинские NER-метки вроде DRUG_BRAND, ADVERSE_EVENT, DOSE. General-purpose models ничего о них не знают. Scispacy и BioBERT - стартовые точки.

## Доставка

Сохраните как `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Упражнения

1. **Легко.** Реализуйте `bio_to_spans` (обратную функцию к `spans_to_bio`) и проверьте round-trip consistency на 10 предложениях.
2. **Средне.** Обучите CRF из sklearn-crfsuite выше на английском NER-датасете CoNLL-2003. Сообщите per-entity F1 с помощью `seqeval`. Типичный результат: ~84 F1.
3. **Сложно.** Fine-tune `distilbert-base-cased` на domain-specific NER dataset (medical, legal или financial). Сравните с малой моделью spaCy. Задокументируйте проверки data leakage и опишите, что вас удивило.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-------------------|--------------------------------|
| NER | Извлекать имена | Размечать token spans типами (PERSON, ORG, GPE, DATE, ...). |
| BIO | Схема тегирования | `B-X` начинает, `I-X` продолжает, `O` вне сущности. |
| BILOU | Улучшенный BIO | Добавляет `L-X` (last), `U-X` (unit) для более чистых границ. |
| CRF | Структурный классификатор | Моделирует переходы между метками, а не только эмиссии. Обеспечивает валидные последовательности. |
| Nested NER | Перекрывающиеся сущности | Один span является другой сущностью, чем его sub-span. BIO не может это выразить. |
| Entity-level F1 | Правильная NER-метрика | Предсказанный span должен точно совпасть с истинным span. Token-level F1 завышает точность. |

## Дополнительное чтение

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — статья о BiLSTM-CRF. Каноническая.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — вводит token-classification pattern, ставший стандартом.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — практический справочник по каждому атрибуту `Doc.ents` и `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — правильная библиотека метрик. Используйте всегда.
