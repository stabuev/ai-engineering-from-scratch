# BERT — masked language modeling

> GPT предсказывает следующее слово. BERT предсказывает пропущенное слово. Разница в одно предложение — и полдесятилетия всего, что похоже на embeddings.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 7 · 05 (полный transformer), Фаза 5 · 02 (представление текста)
**Время:** ~45 минут

## Проблема

В 2018 году каждая NLP-задача — sentiment, NER, QA, entailment — обучала свою модель с нуля на своих размеченных данных. Не было заранее обученного checkpoint, который "понимает английский" и который можно fine-tune. ELMo (2018) показал, что contextual embeddings можно pre-train с bidirectional LSTM; это помогало, но плохо обобщалось.

BERT (Devlin et al. 2018) спросил: что если взять transformer encoder, обучить его на всех предложениях интернета и заставить предсказывать пропущенные слова по контексту с обеих сторон? Потом на downstream task fine-tune одну head. Parameter efficiency стала откровением.

Итог: за 18 месяцев BERT и варианты (RoBERTa, ALBERT, ELECTRA) доминировали во всех NLP leaderboard. К 2020 году BERT был внутри каждого search engine, content moderation pipeline и semantic-search system.

В 2026 году encoder-only модели все еще правильный инструмент для classification, retrieval и structured extraction: они работают на 5–10× быстрее на токен, чем decoders, а их embeddings — backbone современных retrieval stacks. ModernBERT (Dec 2024) довел архитектуру до 8K context с Flash Attention + RoPE + GeGLU.

## Концепция

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### Обучающий сигнал

Возьмите предложение: `the quick brown fox jumps over the lazy dog`.

Случайно замаскируйте 15% токенов:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Обучайте модель предсказывать исходные токены в masked positions. Поскольку encoder bidirectional, предсказание `[MASK]` в позиции 1 может использовать `brown fox jumps` в позициях 2+. Именно этого GPT делать не может.

### Правила маскирования BERT

Из 15% токенов, выбранных для prediction:

- 80% заменяются на `[MASK]`.
- 10% заменяются случайным токеном.
- 10% остаются без изменений.

Почему не всегда `[MASK]`? Потому что `[MASK]` не появляется на inference. Если обучить модель ожидать `[MASK]` в 100% masked positions, возникнет distribution shift между pretraining и fine-tuning. 10% random + 10% unchanged удерживают модель в реальном режиме.

### Next Sentence Prediction (NSP) — и почему от него отказались

Оригинальный BERT также обучался на NSP: по двум предложениям A и B предсказать, следует ли B за A. RoBERTa (2019) провела ablation и показала, что NSP вредил, а не помогал. Современные encoders его пропускают.

### Что изменилось в 2026 году: ModernBERT

Статья ModernBERT 2024 года пересобрала block на primitives 2026 года:

| Компонент | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Чередование local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

И, в отличие от стека 2018 года, он Flash-Attention-native. Inference в 2–3× быстрее на sequence length 8K, чем DeBERTa-v3, при лучших GLUE scores.

### Сценарии, где encoder все еще выбирают в 2026 году

| Задача | Почему encoder выигрывает у decoder |
|--------|--------------------------------------|
| Retrieval / semantic-search embeddings | Bidirectional context = лучшее качество embedding на токен |
| Classification (sentiment, intent, toxicity) | Один forward pass; нет generation overhead |
| NER / token labeling | Output по каждой позиции, нативно bidirectional |
| Zero-shot entailment (NLI) | Classifier head поверх encoder |
| Reranker для RAG | Cross-encoder scoring, в 10x быстрее LLM rerankers |

## Соберите это

### Шаг 1: логика маскирования

См. `code/main.py`. Функция `create_mlm_batch` принимает список token IDs, vocab size и mask probability. Возвращает input IDs (с примененными masks) и labels (только в masked positions, `-100` в остальных местах — PyTorch ignore index).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Шаг 2: запустите MLM prediction на крошечном корпусе

Обучите 2-layer encoder + MLM head на vocabulary из 20 слов и 200 предложениях. Без gradient — делаем forward-pass sanity checks. Полное обучение требует PyTorch.

### Шаг 3: сравните типы масок

Покажите, как three-way rule сохраняет модель usable без `[MASK]`. Предскажите на unmasked sentence и masked sentence. В обоих случаях distributions должны быть разумными, потому что модель видела оба паттерна при training.

### Шаг 4: fine-tune head

Замените MLM head на classification head для toy sentiment dataset. Обучается только head; encoder frozen. Это паттерн каждого BERT-приложения.

## Используйте это

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding-модели — это fine-tuned BERT.** Модели `sentence-transformers`, например `all-MiniLM-L6-v2`, — это BERT, обученные contrastive loss. Encoder тот же. Изменился loss.

**Cross-encoder rerankers — тоже fine-tuned BERT.** Pair-classification на `[CLS] query [SEP] doc [SEP]`. Bidirectional attention между query и doc дает cross-encoders преимущество качества над biencoders.

**Когда не выбирать BERT в 2026 году.** Все generative-задачи. Encoder не имеет разумного способа autoregressively производить токены. Также: задачи до 1B params, где маленький decoder может дать то же качество с большей гибкостью (Phi-3-Mini, Qwen2-1.5B).

## Доведите до поставки

См. `outputs/skill-bert-finetuner.md`. Skill задает scope BERT fine-tune (backbone choice, head spec, data, eval, stopping) для новой classification или extraction task.

## Упражнения

1. **Легко.** Запустите `code/main.py` и напечатайте mask distribution по 10,000 токенов. Подтвердите, что ~15% выбраны, а из них ~80% становятся `[MASK]`.
2. **Средне.** Реализуйте whole-word masking: если слово tokenized into subwords, маскируйте все subwords вместе или ни одно. Измерьте, улучшает ли это MLM accuracy на корпусе из 500 предложений.
3. **Сложно.** Обучите tiny (2-layer, d=64) BERT на 10,000 предложениях из public dataset. Fine-tune токен `[CLS]` для SST-2 sentiment. Сравните с decoder-only baseline при matched params — кто выигрывает?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|------------|-------------------------------|
| MLM | "Masked language modeling" | Training signal: случайно заменить 15% токенов на `[MASK]`, предсказать оригиналы. |
| Bidirectional | "Смотрит в обе стороны" | Encoder attention не имеет causal mask — каждая позиция видит каждую другую. |
| `[CLS]` | "Pooler token" | Special token в начале каждой sequence; его final embedding используется как sentence-level representation. |
| `[SEP]` | "Разделитель сегментов" | Разделяет paired sequences (например, query/doc, sentence A/B). |
| NSP | "Предсказание следующего предложения" | Вторая pretraining task BERT; RoBERTa показала ее бесполезность, после 2019 ее отбросили. |
| Fine-tuning | "Адаптировать к задаче" | Оставить encoder mostly frozen; обучить маленькую head сверху для downstream task. |
| Cross-encoder | "Reranker" | BERT, который принимает query и doc как input и выдает relevance score. |
| ModernBERT | "Обновление 2024 года" | Encoder, пересобранный с RoPE, RMSNorm, GeGLU, alternating local/global attention, 8K context. |

## Дополнительное чтение

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — оригинальная статья.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — как правильно обучать BERT; убирает NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — replaced-token detection лучше MLM при matched compute.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — статья ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — canonical encoder reference.
