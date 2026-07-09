# POS-разметка и синтаксический разбор

> Грамматика какое-то время была не в моде. Затем каждому LLM-пайплайну понадобилось валидировать structured extraction, и она вернулась.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Время:** ~45 минут

## Цели обучения

- Строить baseline most-frequent-tag и биграммный HMM-теггер с декодированием Витерби.
- Набрасывать парсинг зависимостей и схему Universal Dependencies.
- Говорить, где POS-теггинг и парсинг все еще важны в 2026.

## Проблема

Урок 01 обещал, что лемматизации нужна part-of-speech tag. Не зная, что `running` - это глагол, лемматизатор не сможет свести его к `run`. Не зная, что `better` - это прилагательное, он не сможет свести его к `good`.

За этим обещанием скрывалась целая подобласть. Part-of-speech tagging назначает грамматические категории. Syntactic parsing восстанавливает древовидную структуру предложения: какое слово что модифицирует, какой глагол управляет какими аргументами. Классический NLP двадцать лет улучшал и то, и другое. Затем deep learning свел их к token-classification task поверх pretrained transformer, и исследовательское сообщество двинулось дальше.

Но не прикладное сообщество. Каждый structured-extraction pipeline все еще использует POS и dependency trees под капотом. LLM-generated JSON валидируется по грамматическим ограничениям. Question-answering systems декомпозируют запросы с помощью dependency parses. Оценщики качества machine translation проверяют выравнивание parse trees.

Это стоит знать. Этот урок вводит tagsets, baselines и точку, где вы прекращаете реализовывать с нуля и вызываете spaCy.

## Концепция

![Пример POS-тега и dependency-разбора](./assets/pos-parse.svg)

**POS tagging** помечает каждый токен грамматической категорией. **Penn Treebank (PTB)** tagset - английский стандарт. 36 тегов с различиями, которые обычному читателю кажутся придирчивыми: `NN` singular noun, `NNS` plural noun, `NNP` proper noun singular, `VBD` verb past tense, `VBZ` verb 3rd person singular present и так далее. **Universal Dependencies (UD)** tagset грубее (17 тегов) и не привязан к языку; он стал стандартом для cross-lingual work.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Syntactic parsing** строит дерево. Два основных стиля:

- **Constituency parsing.** Noun phrases, verb phrases, prepositional phrases вкладываются друг в друга. Выход - дерево non-terminal categories (NP, VP, PP), где слова являются листьями.
- **Dependency parsing.** Каждое слово имеет одно head word, от которого зависит, с меткой грамматического отношения. Выход - дерево, где каждое ребро является тройкой (head, dependent, relation).

Dependency parsing победил в 2010-х, потому что хорошо обобщается между языками, особенно языками со свободным порядком слов.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Сборка

### Шаг 1: most-frequent-tag baseline

Самый простой POS tagger, который работает. Для каждого слова предсказывайте тег, который оно чаще всего имело в training.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

На Brown corpus этот baseline достигает ~85% accuracy. Не хорошо, но это нижняя граница, ниже которой не должна падать ни одна серьезная модель.

### Шаг 2: bigram HMM tagger

Моделируем совместную вероятность последовательности:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Две таблицы: transition probabilities (tag при заданном previous tag), emission probabilities (word при заданном tag). Оценивайте обе по counts с Laplace smoothing. Декодируйте с Viterbi (dynamic programming по lattice тегов).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM на Brown достигает ~93% accuracy. Скачок с 85% до 93% в основном дают transition probabilities - модель узнает, что `DET NOUN` часто, а `NOUN DET` редко.

### Шаг 3: почему современные taggers лучше

Transition + emission probabilities локальны. Они не могут уловить, что `saw` - существительное в "I bought a saw", но глагол в "I saw the movie." CRF с произвольными признаками (suffix, word shape, слово до и после, само слово) достигает ~97%. BiLSTM-CRF или transformer достигают ~98%+.

Потолок этой задачи задается расхождением аннотаторов. Human annotators согласны примерно в 97% случаев на Penn Treebank. Модели выше 98%, вероятно, overfit test set.

### Шаг 4: набросок dependency parsing

Полный dependency parsing с нуля выходит за рамки; каноническое изложение есть у Jurafsky and Martin. Две классические семьи, которые нужно знать:

- **Transition-based** parsers (arc-eager, arc-standard) работают как shift-reduce parser: читают токены, перекладывают их на stack и применяют reduce actions, создающие arcs. Greedy decoding быстрый. Классическая реализация - MaltParser. Современная нейронная версия: transition-based parser Chen and Manning.
- **Graph-based** parsers (Eisner's algorithm, Dozat-Manning biaffine) оценивают каждое возможное head-dependent edge и выбирают maximum spanning tree. Медленнее, но точнее.

Для большинства прикладных задач вызывайте spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Читайте колонку `dep` снизу вверх, и грамматическая структура предложения проявится сама.

## Использование

Каждая production NLP library поставляет POS и dependency parsers как часть стандартного pipeline.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Быстрая, точная, интегрирована с tokenization + NER + lemmatization. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (dependency relation).
- **Stanford NLP (stanza)**. Преемник Stanford для CoreNLP. State-of-the-art на 60+ языках.
- **trankit**. Transformer-based, хорошая UD accuracy.
- **NLTK**. `pos_tag`. Работает, медленная, старая. Подходит для обучения.

### Почему это все еще важно в 2026 году

- **Lemmatization.** Уроку 01 нужен POS для корректной лемматизации. Всегда.
- **Structured extraction from LLM outputs.** Валидировать, что сгенерированное предложение соблюдает грамматические ограничения (например, subject-verb agreement, required modifiers).
- **Aspect-based sentiment.** Dependency parses показывают, какое прилагательное модифицирует какое существительное.
- **Query understanding.** "movies directed by Wes Anderson starring Bill Murray" раскладывается в structured constraints через parse.
- **Cross-lingual transfer.** UD tags и dependency relations не зависят от языка, что позволяет zero-shot structured analysis новых языков.
- **Low-compute pipelines.** Если вы не можете ship transformer, POS + dependency parse + gazetteer продвинут вас удивительно далеко.

## Доставка

Сохраните как `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Упражнения

1. **Легко.** Используя most-frequent-tag baseline на небольшом tagged corpus (например, NLTK's Brown subset), измерьте accuracy на held-out sentences. Проверьте результат ~85%.
2. **Средне.** Обучите bigram HMM выше и сообщите per-tag precision/recall. Какие tags HMM путает чаще всего?
3. **Сложно.** Используйте dependency parse spaCy, чтобы извлечь subject-verb-object triples из выборки в 1000 предложений. Оцените на 50 manually labeled triples. Задокументируйте, где extraction ломается (часто passives, coordinations и elided subjects).

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-------------------|--------------------------------|
| POS tag | Тип слова | Грамматическая категория. В PTB 36; в UD 17. |
| Penn Treebank | Стандартный tagset | Специфичен для английского. Детализированные времена глаголов и число существительных. |
| Universal Dependencies | Multilingual tagset | Грубее PTB; language-neutral; стандарт для cross-lingual work. |
| Dependency parse | Дерево предложения | Каждое слово имеет один head, каждое ребро имеет грамматическое отношение. |
| Viterbi | Dynamic programming | Находит наиболее вероятную последовательность тегов по emissions и transitions. |

## Дополнительное чтение

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — каноническое учебное изложение POS и parsing.
- [Universal Dependencies project](https://universaldependencies.org/) — cross-lingual tagset и коллекция treebanks, используемые каждым multilingual parser.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — практический справочник по каждому атрибуту, доступному на `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — статья, которая ввела neural parsers в mainstream.
