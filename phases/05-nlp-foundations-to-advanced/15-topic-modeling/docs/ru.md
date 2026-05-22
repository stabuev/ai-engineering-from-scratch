# Topic Modeling — LDA и BERTopic

> LDA: документы — смеси тем, темы — распределения по словам. BERTopic: документы кластеризуются в embedding space, кластеры — это темы. Та же цель, другие примитивы.

**Тип:** Learn
**Языки:** Python
**Предварительные требования:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Время:** ~45 минут

## Проблема

У вас есть 10 000 тикетов поддержки клиентов, 50 000 новостных статей или 200 000 твитов. Вам нужно понять, о чем коллекция, не читая ее. У вас нет размеченных категорий. Вы даже не знаете, сколько категорий существует.

Topic modeling отвечает на это без supervision. Дайте ему корпус — получите небольшой набор связных тем и, для каждого документа, распределение по этим темам.

Доминируют две алгоритмические семьи. LDA (2003) рассматривает каждый документ как смесь latent topics, а каждую тему — как распределение по словам. Inference — байесовский. Он все еще поставляется в продакшен там, где нужны mixed-membership topic assignments и объяснимые распределения вероятностей на уровне слов.

BERTopic (2020) кодирует документы с BERT, снижает размерность с UMAP, кластеризует с HDBSCAN и извлекает topic words через class-based TF-IDF. Он выигрывает на коротком тексте, социальных медиа и всем, где семантическая близость важнее пересечения слов. Один документ получает одну тему, что является ограничением для long-form content.

Этот урок строит интуицию для обоих подходов и называет, какой выбрать для заданного корпуса.

## Концепция

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**Генеративная история LDA.** Каждая тема — распределение по словам. Каждый документ — смесь тем. Чтобы сгенерировать слово в документе, сэмплируем тему из смеси документа, затем сэмплируем слово из распределения этой темы. Inference обращает это: по наблюдаемым словам выводит topic distribution для каждого документа и word distribution для каждой темы. Collapsed Gibbs sampling или variational Bayes выполняет математику.

Ключевой output LDA:

- `doc_topic`: матрица `(n_docs, n_topics)`, каждая строка суммируется в 1 (topic mixture документа).
- `topic_word`: матрица `(n_topics, vocab_size)`, каждая строка суммируется в 1 (word distribution темы).

**Pipeline BERTopic.**

1. Кодируйте каждый документ sentence transformer (например, `all-MiniLM-L6-v2`). 384-мерные векторы.
2. Снизьте размерность с UMAP примерно до 5 измерений. BERT embeddings слишком высокоразмерны для clustering.
3. Кластеризуйте с HDBSCAN. Density-based, создает кластеры переменного размера и label "outlier".
4. Для каждого кластера вычислите class-based TF-IDF по документам кластера, чтобы извлечь top words.

Output — одна тема на документ (плюс label выброса -1). Опционально — soft membership через probability vector HDBSCAN.

## Собираем

### Шаг 1: LDA через scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Обратите внимание: stopwords удалены, min_df и max_df фильтруют редкие и повсеместные термины, CountVectorizer (не TfidfVectorizer), потому что LDA ожидает raw counts.

### Шаг 2: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Фильтр по `Topic != -1` удаляет BERTopic's outlier bucket (документы, которые HDBSCAN не смог кластеризовать). `min_topic_size` управляет минимальным размером кластера HDBSCAN; значение по умолчанию в библиотеке BERTopic — 10. В этом примере оно явно установлено в 15 для масштаба урока. Для корпусов свыше 10 000 документов увеличьте до 50 или 100.

### Шаг 3: оценка

Оба метода выводят topic words. Вопрос в том, насколько эти слова связны.

- **Topic coherence (c_v).** Объединяет NPMI (normalized pointwise mutual information) пар top-word по sliding-window contexts, агрегирует scores в topic vectors и сравнивает эти векторы через cosine similarity. Чем выше, тем лучше. Используйте `gensim.models.CoherenceModel` с `coherence="c_v"`.
- **Topic diversity.** Доля уникальных слов среди top words всех тем. Чем выше, тем лучше (темы не пересекаются).
- **Qualitative inspection.** Прочитайте top words каждой темы. Называют ли они реальную вещь? Человеческое суждение все еще последняя линия защиты.

## Когда что выбирать

| Situation | Pick |
|-----------|------|
| Short text (tweets, reviews, headlines) | BERTopic |
| Long documents with topic mixtures | LDA |
| No GPU / limited compute | LDA or NMF |
| Need document-level multi-topic distributions | LDA |
| LLM integration for topic labeling | BERTopic (direct support) |
| Resource-constrained edge deployment | LDA |
| Max semantic coherence | BERTopic |

Самое важное практическое соображение — длина документа. BERT embeddings усекаются; LDA counts работают с любой длиной. Для документов длиннее контекста embedding model либо chunk + aggregate, либо используйте LDA.

## Применение

Стек 2026 года:

- **BERTopic.** Вариант по умолчанию для короткого текста и всего, где важна семантика.
- **`gensim.models.LdaModel`.** Классический LDA для production, зрелый, проверенный.
- **`sklearn.decomposition.LatentDirichletAllocation`.** Простой LDA для экспериментов.
- **NMF.** Non-negative matrix factorization. Быстрая альтернатива LDA, сопоставимое качество на коротком тексте.
- **Top2Vec.** Похожий на BERTopic дизайн. Меньшее сообщество, но хорош на некоторых benchmarks.
- **FASTopic.** Новее, быстрее BERTopic на очень больших корпусах.
- **LLM-based labeling.** Запустите любую кластеризацию, затем попросите model назвать каждый кластер.

## Доставка

Сохраните как `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Упражнения

1. **Easy.** Обучите LDA с 5 темами на dataset 20 Newsgroups. Выведите top 10 words для каждой темы. Подпишите каждую тему вручную. Нашел ли алгоритм реальные категории?
2. **Medium.** Обучите BERTopic на том же subset 20 Newsgroups. Сравните найденное число тем, top words и qualitative coherence с LDA. Что чище выявляет реальные категории?
3. **Hard.** Вычислите c_v coherence для LDA и BERTopic на вашем корпусе. Запустите каждый метод с 5, 10, 20, 50 темами. Постройте график coherence vs topic count. Сообщите, какой метод стабильнее по разным числам тем.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Topic | A thing the corpus is about | Распределение вероятностей по словам (LDA) или кластер похожих документов (BERTopic). |
| Mixed membership | Doc is multiple topics | LDA назначает каждому документу распределение по всем темам. |
| UMAP | Dimensionality reduction | Manifold learning, сохраняющий локальную структуру; используется в BERTopic. |
| HDBSCAN | Density clustering | Находит кластеры переменного размера; создает label "noise" (-1) для выбросов. |
| c_v coherence | Topic quality metric | Средняя pointwise mutual information top topic words внутри sliding windows. |

## Дополнительное чтение

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — статья LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — статья BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — статья, представившая c_v и родственные меры.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — reference для production. Отличные примеры.
