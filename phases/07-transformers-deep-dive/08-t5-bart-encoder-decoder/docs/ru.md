# T5, BART — encoder-decoder модели

> Encoders понимают. Decoders генерируют. Соедините их обратно — и получите модель для задач input → output: translate, summarize, rewrite, transcribe.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 7 · 05 (полный transformer), Фаза 7 · 06 (BERT), Фаза 7 · 07 (GPT)
**Время:** ~45 минут

## Проблема

Decoder-only GPT и encoder-only BERT упрощают архитектуру 2017 года под разные цели. Но многие задачи естественно являются input-output:

- Translation: English → French.
- Summarization: 5,000-token article → 200-token summary.
- Speech recognition: audio tokens → text tokens.
- Structured extraction: prose → JSON.

Для них encoder-decoder подходит чище всего. Encoder производит плотное представление source. Decoder генерирует output, cross-attending к этому представлению на каждом step. Training — shift-by-one на output side. Тот же loss, что у GPT, только conditioned on encoder output.

Две статьи задали современный playbook:

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." Каждая NLP task переоформлена как text-in, text-out. Одна architecture, один vocabulary, один loss. Pretrained на masked span prediction (corrupt spans во input, decode them in output).
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." Denoising autoencoder: corrupt input разными способами (shuffle, mask, delete, rotate), попросить decoder восстановить original.

В 2026 году формат encoder-decoder живет там, где важна структура input:

- Whisper (speech → text).
- Translation stack Google.
- Некоторые code-completion / repair models с отдельными context-and-edit structures.
- Flan-T5 и варианты для structured reasoning tasks.

Decoder-only забрал внимание, но encoder-decoder не исчез.

## Концепция

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### Forward loop

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Ключевое: encoder запускается один раз на input. Decoder работает autoregressively, но cross-attends к одному и тому же encoder output на каждом step. Caching encoder output — бесплатное ускорение для long inputs.

### Pretraining T5 — span corruption

Выберите random spans во input (средняя длина 3 tokens, 15% total). Замените каждый span уникальным sentinel: `<extra_id_0>`, `<extra_id_1>` и т.д. Decoder выводит только corrupted spans с их sentinel prefix:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Это более дешевый signal, чем предсказывать всю sequence. В ablation T5 он конкурентен с MLM (BERT) и prefix-LM (UniLM).

### Pretraining BART — denoising с несколькими видами шума

BART пробует пять noising functions:

1. Token masking.
2. Token deletion.
3. Text infilling (mask span, decoder вставляет правильную длину).
4. Sentence permutation.
5. Document rotation.

Комбинация text infilling + sentence permutation дала лучшие downstream numbers. Decoder всегда reconstructs original. Output BART — full sequence, не только corrupted spans, поэтому pretraining compute выше, чем у T5.

### Inference

Та же autoregressive generation, что у GPT. Greedy / beam / top-p sampling применимы. Beam search (width 4–5) стандартен для translation и summarization, потому что output distribution уже, чем в chat.

### Когда выбирать вариант в 2026 году

| Задача | Encoder-decoder? | Почему |
|--------|------------------|--------|
| Translation | Да, обычно | Четкая source sequence; fixed output distribution; beam search работает |
| Speech-to-text | Да (Whisper) | Input modality отличается от output; encoder формирует audio features |
| Chat / reasoning | Нет, decoder-only | Нет постоянного "input" — conversation и есть sequence |
| Code completion | Обычно нет | Decoder-only с long context выигрывает; code models вроде Qwen 2.5 Coder — decoder-only |
| Summarization | Подходит любой | BART, PEGASUS били ранние decoder-only baselines; современные decoder-only LLMs сравнялись с ними |
| Structured extraction | Любой | T5 удобен, потому что "text → text" поглощает любой output format |

Тренд с ~2022: decoder-only забирает задачи, где раньше был encoder-decoder, потому что (a) instruction-tuned decoder-only LLMs обобщаются на все через prompting, (b) одну architecture проще масштабировать, чем две, (c) RLHF предполагает decoder. Encoder-decoder держится там, где input modality отличается (speech, images) или важен beam search quality.

## Соберите это

См. `code/main.py`. Мы реализуем T5-style span corruption для toy corpus — это самая полезная часть урока, потому что она встречается почти во всех encoder-decoder pretraining recipes.

### Шаг 1: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Target format — convention T5: `<sent0> span0 <sent1> span1 ...`. Corrupted input чередует unchanged tokens с sentinel tokens в местах spans.

### Шаг 2: проверьте round-trip

По corrupted input и target восстановите original sentence. Если corruption обратим, forward pass определен корректно. Это sanity check — real training так не делает, но тест дешевый и ловит off-by-one bugs в span bookkeeping.

### Шаг 3: noising в BART

Пять функций: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Скомпонуйте две и покажите результат.

## Используйте это

Reference в HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Трюк T5: имя task входит в input text. Одна model обрабатывает десятки tasks, потому что каждая task — text-in, text-out. В 2026 году этот pattern обобщен instruction-tuned decoder-only models, но T5 зафиксировал его первым.

## Доведите до поставки

См. `outputs/skill-seq2seq-picker.md`. Skill выбирает между encoder-decoder и decoder-only для новой task по input-output structure, latency и quality targets.

## Упражнения

1. **Легко.** Запустите `code/main.py`, примените span corruption к 30-token sentence, проверьте, что concatenating non-sentinel source tokens с decoded target spans воспроизводит original.
2. **Средне.** Реализуйте BART `text_infill` noise: замените random spans одним `<mask>` token, а decoder должен вывести правильную длину и содержимое span. Покажите пример.
3. **Сложно.** Fine-tune `flan-t5-small` на tiny English → pig-Latin corpus (200 pairs). Измерьте BLEU на held-out 50-pair set. Сравните с fine-tuning `Llama-3.2-1B` на тех же data с тем же compute.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|------------|-------------------------------|
| Encoder-decoder | "Seq2seq transformer" | Два стека: bidirectional encoder для input, causal decoder с cross-attention для output. |
| Cross-attention | "Где source говорит с target" | Decoder Q × encoder K/V. Единственное место, где encoder information входит в decoder. |
| Span corruption | "Pretraining-трюк T5" | Заменить random spans на sentinel tokens; decoder outputs spans. |
| Denoising objective | "Игра BART" | Применить noise function к input, обучить decoder reconstruct clean sequence. |
| Sentinel token | "Placeholder `<extra_id_N>`" | Special tokens, которые отмечают corrupted spans в source и повторно отмечают их в target. |
| Flan | "Instruction-tuned T5" | T5 fine-tuned на >1,800 tasks; сделал encoder-decoder конкурентным в instruction-following. |
| Beam search | "Стратегия decoding" | Держать top-k partial sequences на каждом step; стандарт для translation/summarization. |
| Teacher forcing | "Вход во время обучения" | Во время training подавать true previous output token в decoder, а не sampled one. |

## Дополнительное чтение

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, canonical encoder-decoder 2026 года.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — reference implementation.
