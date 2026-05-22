# Извлечение отношений и построение графов знаний

> NER нашла сущности. Связывание сущностей закрепило их. Извлечение отношений находит ребра между ними. Граф знаний - это сумма узлов, ребер и их происхождения.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 5 · 06 (NER), Фаза 5 · 25 (Entity Linking)
**Время:** ~60 минут

## Проблема

Аналитик читает: "Tim Cook became CEO of Apple in 2011." Четыре факта:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Извлечение отношений (Relation Extraction, RE) превращает свободный текст в структурированные тройки `(subject, relation, object)`. Соберите их по корпусу - и получите граф знаний. Соберите, затем запрашивайте - и получите основу для рассуждений в RAG, аналитике или аудитах соответствия.

Проблема 2026 года: LLM извлекают отношения с энтузиазмом. Слишком большим энтузиазмом. Они галлюцинируют тройки, которые исходный текст не подтверждает. Без происхождения данных вы не можете отличить реальные тройки от правдоподобной выдумки. Ответ 2026 года - конвейеры в стиле AEVS: закрепить и проверить.

## Концепция

![Текст → тройки → граф знаний](../assets/relation-extraction.svg)

**Форма тройки.** `(subject_entity, relation_type, object_entity)`. Отношения берутся из закрытой онтологии (свойства Wikidata, FIBO, UMLS) или из открытого множества (стиль OpenIE: допустимо что угодно).

**Три подхода к извлечению.**

1. **На правилах / шаблонах.** Шаблоны Hearst: "X such as Y" → `(Y, isA, X)`. Плюс вручную написанные regex. Хрупко, точно, объяснимо.
2. **Обучаемый классификатор.** Для двух упоминаний сущностей в предложении предсказать отношение из фиксированного набора. Обучается на TACRED, ACE, KBP. Стандарт 2015-2022.
3. **Генеративная LLM.** Запросите модель, чтобы она выдала тройки. Работает сразу. Нуждается в происхождении данных, иначе галлюцинирует правдоподобный мусор.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Текущий фреймворк для снижения галлюцинаций:

- **Anchor.** Определить каждый span сущности и span фразы отношения с точными позициями.
- **Extract.** Сгенерировать тройки, связанные с anchor spans.
- **Verify.** Сопоставить каждый элемент тройки с исходным текстом; отклонить все неподтвержденное.
- **Supplement.** Проход по полноте гарантирует, что ни один закрепленный span не пропущен.

Галлюцинации резко снижаются. Требует больше вычислений, но пригодно для аудита.

**Компромисс open-vs-closed.**

- **Закрытая онтология.** Фиксированный список свойств (например, 11 000+ свойств Wikidata). Предсказуемо. Запрашиваемо. Трудно выдумывать.
- **Open IE.** Любая глагольная фраза становится отношением. Высокая полнота. Низкая точность. Сложно запрашивать.

Производственные графы знаний обычно смешивают подходы: Open IE для обнаружения, затем канонизируют отношения в закрытую онтологию перед слиянием с основным графом.

## Соберите это

### Шаг 1: извлечение на шаблонах

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Полный игрушечный извлекатель см. в `code/main.py`. Шаблоны Hearst все еще поставляют в предметно-специфических конвейерах, потому что их можно отлаживать.

### Шаг 2: обучаемая классификация отношений

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL - seq2seq-извлекатель отношений: на вход текст, на выход тройки, уже в идентификаторах свойств Wikidata. Дообучен на данных distant supervision. Стандартная baseline-модель с открытыми весами.

### Шаг 3: извлечение через LLM-промпт с anchoring

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Проверяйте каждый возвращенный span по источнику. Отклоняйте все, где `text[start:end] != triple_entity`. Это шаг AEVS "verify" в минимальной форме.

### Шаг 4: канонизировать в закрытую онтологию

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Канонизация часто составляет 60-80% инженерной работы. Заложите на нее бюджет.

### Шаг 5: построить небольшой граф и выполнить запрос

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Это атом каждой системы RAG-over-KG. Масштабируйте его с помощью RDF-хранилищ троек (Blazegraph, Virtuoso), property graph (Neo4j) или графовых хранилищ, дополненных векторами.

## Подводные камни

- **Кореференция перед RE.** "He founded Apple" — RE должно знать, кто такой "he". Сначала запускайте coref (урок 24).
- **Канонизация сущностей.** "Apple Inc" и "Apple" должны разрешаться в один и тот же узел. Сначала entity linking (урок 25).
- **Галлюцинированные тройки.** LLM выдают тройки, которые текст не подтверждает. Принудительно применяйте проверку span.
- **Дрейф канонизации отношений.** Отношения Open IE непоследовательны ("was born in," "came from," "is a native of"). Сводите к каноническим id, иначе граф нельзя будет запрашивать.
- **Темпоральные ошибки.** "Tim Cook is CEO of Apple" — верно сейчас, неверно в 2005 году. Многие отношения ограничены во времени. Используйте qualifiers (`P580` start time, `P582` end time in Wikidata).
- **Несоответствие домена.** REBEL обучен на Wikipedia. Юридические, медицинские и научные тексты часто требуют RE-моделей, дообученных под домен.

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Быстрый production, общий домен | REBEL или LlamaPred с канонизацией Wikidata |
| Предметно-специфично (биомедицина, право) | Дообучение в стиле SciREX + пользовательская онтология |
| LLM-промптинг, аудируемый вывод | Конвейер AEVS: anchor → extract → verify → supplement |
| Высокообъемное IE новостей | Гибрид на шаблонах + обучаемой модели |
| Построение KG с нуля | Open IE + ручной проход канонизации |
| Темпоральный KG | Извлекать с qualifiers (start/end time, point in time) |

Шаблон интеграции: NER → coref → entity linking → relation extraction → ontology mapping → graph load. Каждый этап - потенциальный quality gate.

## Доведите до поставки

Сохраните как `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Упражнения

1. **Легко.** Запустите шаблонный извлекатель из `code/main.py` на 5 предложениях из новостных статей. Вручную проверьте precision.
2. **Средне.** Используйте REBEL (или небольшую LLM) на тех же предложениях. Сравните тройки. У какого извлекателя выше precision? Выше recall?
3. **Сложно.** Постройте конвейер AEVS: извлечение с LLM + проверка spans по источнику. Измерьте hallucination rate до и после шага verify на 50 предложениях в стиле Wikipedia.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | Кортеж `(s, r, o)`, который является атомарной единицей KG. |
| Open IE | Extract anything | Фразы отношений с открытым словарем; высокая полнота, низкая точность. |
| Closed ontology | Fixed schema | Ограниченный набор типов отношений (Wikidata, UMLS, FIBO). |
| Canonicalization | Normalize everything | Отображение поверхностных имен / отношений в канонические ids. |
| AEVS | Grounded extraction | Конвейер Anchor-Extraction-Verification-Supplement (2026). |
| Provenance | Source-of-truth link | Каждая тройка несет doc id + char-span к своему источнику. |
| Distant supervision | Cheap labels | Сопоставление текста с существующим KG для создания обучающих данных. |

## Дополнительное чтение

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — статья о distant supervision.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — рабочая seq2seq-модель RE.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — совместное IE.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — дизайн снижения галлюцинаций 2026 года.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — канонические графовые запросы.
