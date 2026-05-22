# Differential Attention (V2)

> Softmax-attention распределяет небольшой объем вероятности по каждому несовпадающему токену. На 100k токенов этот шум накапливается и заглушает сигнал. Differential Transformer (Ye et al., ICLR 2025) исправляет это, вычисляя внимание как разность двух softmax, вычитая общий шумовой уровень. DIFF V2 (Microsoft, январь 2026) — переписывание для production-стека: задержка декодирования как у базового Transformer, без кастомных kernels, совместимо с FlashAttention. Этот урок проходит путь от V1 к V2 end-to-end, с рабочей toy-реализацией операции разности, которую можно запустить на stdlib Python.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## Learning Objectives

- Точно сформулировать, почему у softmax attention есть noise floor и почему он растет с длиной контекста.
- Вывести формулу differential attention и объяснить, почему вычитание убирает общий шумовой компонент, сохраняя сигнал.
- Разобрать diff от V1 к V2: что стало быстрее, проще и стабильнее, и почему каждое изменение было нужно для production pre-training.
- Реализовать differential attention с нуля на чистом Python и эмпирически проверить свойство шумоподавления на синтетическом запросе signal-plus-noise.

## The Problem

У стандартного softmax attention есть математическое свойство, которое в масштабе становится операционной проблемой. Для запроса `q` веса внимания равны `softmax(qK^T / sqrt(d))`. Softmax никогда не дает точных нулей — каждый несовпадающий токен получает некоторую положительную массу. Эта остаточная масса является шумом, и она масштабируется с длиной контекста. При 128k токенов, даже если каждый несовпадающий токен получает всего 0.001% вероятности, 127,999 таких токенов вместе дают около 12% общей массы. Модель вынуждена учиться обходить noise floor, который растет вместе с контекстом.

Эмпирически это проявляется как interference между attention heads: галлюцинированные цитаты в long-context RAG, провалы lost-in-the-middle на retrieval-задачах в 100k токенов и тонкое ухудшение accuracy на needle-in-haystack benchmark после 32k. В статье Differential Transformer (arXiv:2410.05258, ICLR 2025) измерили этот разрыв: DIFF Transformers достигают меньшей perplexity, более высокой long-context accuracy и меньшего числа галлюцинаций, чем базовые модели того же размера.

У DIFF V1 было три проблемы, которые не пускали его во frontier pre-training pipelines. Value cache нужно было загружать дважды на каждый шаг decode, требовались кастомные CUDA kernels, ломавшие совместимость с FlashAttention, а per-head RMSNorm дестабилизировал долгие training runs в масштабе 70B+. DIFF V2 (Microsoft unilm blog, 20 января 2026) исправил все три. Этот урок разбирает обе версии, строит оператор разности и benchmark-ит шумоподавление на toy-запросе.

## The Concept

### The noise floor of softmax

Для запроса `q` и ключей `K = [k_1, ..., k_N]` веса внимания:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Ни один `w_i` никогда не равен нулю. Если `k_i` полностью не связан с `q`, score `q . k_i` не равен 0 — он флуктуирует около нуля с дисперсией `||q||^2 / d`. После softmax-нормализации каждый нерелевантный токен все равно вносит `O(1/N)` во взвешенную сумму. Суммарный вклад нерелевантных токенов равен `O((N-1)/N) = O(1)` — это не малая величина.

Модель хочет что-то похожее на hard top-k: большой вес на совпадающих токенах и почти нулевой вес везде остальном. Softmax слишком гладкий, чтобы делать это напрямую.

### The differential idea

Разделим Q и K projection каждой head на две части: Q = (Q_1, Q_2) и K = (K_1, K_2). Посчитаем две attention maps:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Output:

```
DiffAttn = (A_1 - lambda * A_2) V
```

Вычитание убирает то распределение шума, которое является общим для двух maps. Если обе maps имеют примерно равномерный вес на 127k нерелевантных токенов (что при случайной инициализации так и будет), эти веса взаимно уничтожаются. Сигнал — пик веса на нескольких действительно релевантных токенах — уничтожится только если он появляется в обеих maps с одинаковой величиной, чего после обучения модели не будет.

`lambda` — обучаемый скаляр на head, параметризованный как `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Он может быть отрицательным. `lambda_init` обычно задают небольшим положительным числом, например 0.8.

### Why this matches headed noise-canceling

Представьте два шумных микрофона, записывающих один и тот же голос. Оба ловят говорящего плюс коррелированный фоновый шум. Вычтите один сигнал из другого, и общий шум уйдет. Голос сохранится, потому что два сигнала достаточно различаются по фазе или амплитуде, чтобы не погаситься полностью. Per-head `lambda` учится именно этому балансу.

### V1 vs V2: the diff

V1 сохранял число параметров равным базовому Transformer. Чтобы получить два query на head, он делил head dimension пополам. Это снижало выразительность head и, что больнее, делило value cache на head пополам. Decode должен был загружать value cache дважды за шаг (по одному разу на каждую softmax-ветку). Итог: decode медленнее baseline, несмотря на равное число параметров.

V2 удваивает число query heads и оставляет KV heads прежними (заимствуя параметры из up-projection). Head dimension остается такой же, как в baseline. После вычитания лишняя размерность проецируется обратно вниз, чтобы совпасть с O_W projection базового Transformer. Одновременно происходят три вещи:

1. Скорость decode совпадает с baseline (KV cache загружается один раз).
2. FlashAttention работает без изменений (нет кастомного kernel).
3. Arithmetic intensity на decode растет (больше compute на каждый byte, загруженный из HBM).

V2 также убирает per-head RMSNorm, который V1 использовал для стабилизации вычитания. В pre-training масштаба 70B этот RMSNorm дестабилизировал поздние этапы обучения. V2 заменяет его более простой схемой инициализации, которая сохраняет training стабильным без дополнительного модуля.

### When to reach for it

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | Более чистые attention maps, меньше галлюцинированных цитат |
| Needle-in-haystack benchmarks | Существенный прирост accuracy после 32k |
| Multi-document QA | Меньше cross-document interference |
| Code completion at 8k | Предельный выигрыш, не стоит смены архитектуры |
| Short chat (< 4k) | Практически неотличимо от baseline |

Ценность растет вместе с длиной контекста. На 4k токенов noise floor достаточно мал, и стандартное attention работает нормально. На 128k он уже вредит.

### How it stacks with other 2026 knobs

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Да (V2 увеличивает Q heads, не KV heads) |
| MLA (DeepSeek) | Да в принципе, опубликованной статьи с их объединением нет |
| MoE | Да (attention независимо от MLP block) |
| RoPE | Да (без изменений) |
| YaRN / long-context scaling | Да (ровно там DIFF помогает сильнее всего) |
| FlashAttention | Да в V2 (нет в V1) |
| Speculative decoding | Да (изменение attention невидимо для spec-decode loop) |

## Build It

`code/main.py` реализует differential attention на чистом Python. Toy-запрос с известной структурой signal-plus-noise позволяет напрямую измерить коэффициент шумоподавления.

### Step 1: standard softmax attention

Stdlib matrix ops: списки списков, ручной matmul, softmax с вычитанием максимума для численной стабильности.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2: split Q, K into two halves

Стиль V1: делим head dimension пополам. Стиль V2: сохраняем head dimension и удваиваем число heads. Toy-реализация использует V1 для педагогической ясности — математика идентична, отличается только bookkeeping.

### Step 3: two softmax branches + subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Примечание: выходные веса могут быть отрицательными. Это нормально — value cache все равно обрабатывает знаковые вклады. Следующая V projection поглощает знак.

### Step 4: noise cancellation measurement

Постройте синтетическую последовательность длины 1024. Поместите signal token в известную позицию, остальное заполните шумом. Посчитайте (a) вес стандартного softmax attention на signal position и (b) вес differential attention. Измерьте signal-to-noise ratio в каждом случае. DIFF attention стабильно дает более высокий signal-to-noise ratio в 3x-10x раз, в зависимости от того, насколько две ветки обучились различаться.

### Step 5: V1 vs V2 parameter accounting

Для конфига (hidden=4096, heads=32, d_head=128) напечатайте:

- Baseline Transformer: Q, K, V каждый размера `hidden * hidden`, MLP равен 4 * hidden.
- DIFF V1: Q, K каждый размера `hidden * hidden`, V размера `hidden * hidden` (без изменений), head dim внутри поделен пополам. Добавляет per-head параметры `lambda` (O(heads * d_head)).
- DIFF V2: Q размера `2 * hidden * hidden`, K размера `hidden * hidden`, V размера `hidden * hidden`. Дополнительная размерность проецируется обратно вниз перед O_W. Добавляет те же параметры `lambda`.

Toy-код измеряет дополнительную стоимость параметров для V2 (примерно `hidden * hidden` сверху на attention block) и печатает ее.

## Use It

DIFF V2 еще не поставляется в каждом production inference server по состоянию на апрель 2026, но интеграция идет в vLLM и SGLang. Тем временем pattern встречается в:

- Внутренних long-context production models Microsoft.
- Research replications в нескольких open model training runs, нацеленных на 256k+ context.
- Hybrid architectures, которые комбинируют DIFF attention со sliding-window attention на чередующихся layers.

Когда вы бы использовали это в 2026:

- Training новой модели с нуля, нацеленной на эффективный контекст 64k+. Добавьте differential attention с самого начала; переобучение позже дорого.
- Fine-tuning long-context model, где lost-in-the-middle failures доминируют eval. LoRA на Q projections может приблизить DIFF-структуру.

Когда не стоит:

- Вы обслуживаете pre-trained dense model со стабильной long-context performance. Стоимость переобучения редко окупается на существующих weights.
- Контекст всегда меньше 16k. Noise floor пренебрежим.

## Ship It

Этот урок создает `outputs/skill-diff-attention-integrator.md`. По architecture модели, целевой длине контекста, профилю hallucination и training budget он строит integration plan для добавления differential attention в новый pre-training run или LoRA fine-tune.

## Exercises

1. Запустите `code/main.py`. Проверьте, что signal-to-noise ratio для differential attention выше, чем для standard softmax attention на синтетическом запросе. Меняйте noise amplitude и покажите crossover point, где standard attention становится непригодным.

2. Посчитайте parameter-count delta от baseline к DIFF V1 и от baseline к DIFF V2 для модели класса 7B (hidden=4096, heads=32, d_head=128, 32 layers). Покажите, какие компоненты получили параметры, а какие остались прежними.

3. Прочитайте Section 3 статьи DIFF V1 (arXiv:2410.05258) и Section 2 блога DIFF V2 на Hugging Face. В двух предложениях объясните, почему per-head RMSNorm в V1 был необходим и почему V2 мог убрать его без divergence обучения.

4. Реализуйте ablation: посчитайте differential attention с `lambda = 0` (чистый первый softmax) и `lambda = 1` (полное вычитание). На синтетическом запросе измерьте, как signal-to-noise меняется по sweep. Найдите `lambda`, максимизирующий signal-to-noise.

5. Расширьте toy до GQA + DIFF V2. Возьмите 8 KV heads и 32 Q heads. Покажите, что размер KV cache совпадает с baseline GQA model с тем же конфигом (8, 32).

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Differential attention | "Two softmaxes minus each other" | Разделить Q, K на две половины, посчитать две softmax maps, вычесть вторую (масштабированную lambda) из первой, затем умножить на V |
| Noise floor | "The non-zero tail of softmax" | Вес O(1/N), который softmax кладет на каждый нерелевантный токен и который суммируется до O(1) на длинных контекстах |
| lambda | "The subtraction scale" | Per-head обучаемый скаляр, параметризованный как `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; может быть отрицательным |
| DIFF V1 | "The ICLR 2025 version" | Исходный Differential Transformer; делит head dim пополам для сохранения числа параметров, требует кастомный kernel, decode медленнее |
| DIFF V2 | "The January 2026 fix" | Удваивает Q heads, сохраняя KV heads; совпадает по скорости decode с baseline и работает с FlashAttention |
| Per-head RMSNorm | "The V1 stabilizer" | Дополнительная norm, которую V1 применял после разности; V2 убрал ее, чтобы предотвратить нестабильность late-training |
| Signal-to-noise ratio | "How much attention is wasted" | Отношение веса на истинной signal position к среднему весу на нерелевантных позициях |
| Lost in the middle | "Long-context failure mode" | Эмпирический феномен, где retrieval accuracy проседает для документов в середине длинного контекста — DIFF attention снижает это |
| Arithmetic intensity | "FLOPs per byte loaded" | Отношение, которое V2 увеличивает на decode, удваивая queries на одну KV load; важно для memory-bound decode |

## Further Reading

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — исходная статья с теорией noise-cancellation и long-context ablations
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — production-stack rewrite, baseline decode, совместимость с FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — теоретический анализ того, почему вычитание восстанавливает pretrained attention structure
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — вариант с parameter sharing
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — baseline Transformer, из которого DIFF вычитает
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — long-context benchmark, на который нацелено DIFF attention
