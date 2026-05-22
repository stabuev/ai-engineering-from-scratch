# Why Transformers — Проблемы RNN

> RNN обрабатывают токены по одному. Transformers обрабатывают все токены сразу. Эта единственная архитектурная ставка изменила все кривые масштабирования в deep learning после 2017 года.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## Проблема

До 2017 года каждая state-of-the-art модель последовательностей на планете — язык, перевод, речь — была рекуррентной нейронной сетью. LSTM и GRU полдесятилетия выигрывали в бенчмарках перевода уровня ImageNet. Это был единственный инструмент, который был у всех.

У них было три критических слабых места. Последовательное вычисление означало, что нельзя распараллелить работу вдоль временной оси: токену `t+1` нужно скрытое состояние от токена `t`. Последовательность из 1,024 токенов означала 1,024 последовательных шага на GPU, который может выполнять 1,000,000 операций с плавающей точкой за цикл. Фактическое время обучения росло линейно с длиной последовательности на железе, спроектированном для параллелизма.

Исчезающие градиенты означали, что информация 50 токенов назад уже была сжата через 50 нелинейностей. Рекуррентные блоки с вентилями (LSTM, GRU) смягчали это сдавливание, но никогда не устраняли его. Дальние зависимости — "the book I read last summer on a plane to Kyoto was…" — регулярно ломались.

Скрытые состояния фиксированной ширины означали, что encoder сжимал всю исходную последовательность в один вектор до того, как decoder видел хоть что-то. Неважно, в источнике 5 токенов или 500; bottleneck имеет одну и ту же форму.

Статья 2017 года "Attention Is All You Need" предложила радикальную идею: полностью отказаться от recurrence. Позволить каждой позиции attend к каждой другой позиции параллельно. Обучать одной большой матричной операцией вместо 1,024 последовательных.

К 2026 году результат доминирует во всех модальностях. Language (GPT-5, Claude 4, Llama 4), vision (ViT, DINOv2, SAM 3), audio (Whisper), biology (AlphaFold 3), robotics (RT-2). Один и тот же блок, разные входы.

## Концепция

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence как bottleneck.** RNN вычисляет `h_t = f(h_{t-1}, x_t)`. Каждый шаг зависит от предыдущего. Нельзя вычислить `h_5` до `h_4`. На современных GPU с 10,000+ параллельными ядрами это впустую тратит 99% кремния на длинной последовательности.

**Attention как широковещательная операция.** Self-attention вычисляет `output_i = sum_j(a_ij * v_j)` для каждой пары `(i, j)` одновременно. Вся N×N матрица attention заполняется одним batched matmul. Ни один шаг не зависит от другого. GPU это любят.

**Ускорение не является константой.** Это разница между `O(N)` последовательной глубиной и `O(1)` последовательной глубиной. На практике transformers обучаются в 5–10× быстрее за эпоху на сопоставимом железе при N=512, и разрыв растет с длиной последовательности, пока вы не упретесь в `O(N²)` стену памяти attention (которую позже исправил Flash Attention — см. Lesson 12).

**Цена transformers.** Память attention масштабируется как `O(N²)`. Для контекста 2K это нормально. Для контекста 128K нужны sliding windows, RoPE extrapolation, Flash Attention tiling или варианты linear attention. Recurrence была `O(N)` и по времени, и по памяти; transformers меняют время на память, а затем возвращают выигрыш во времени за счет параллелизма.

**Сдвиг inductive bias.** RNN предполагают локальность и близость во времени. Transformers ничего не предполагают — каждая пара является кандидатом на attention. Поэтому transformers требуют больше данных для хорошего обучения, но масштабируются дальше, когда эти данные есть. Chinchilla (2022) формализовала это: при достаточном числе токенов transformer всегда превосходит RNN с тем же числом параметров.

## Build It

Никакой нейронной сети здесь нет — мы численно симулируем ключевой bottleneck, чтобы вы почувствовали разрыв на своем ноутбуке.

### Step 1: измерьте последовательную глубину

См. `code/main.py`. Мы строим две функции. Одна кодирует последовательность как цепочку сложений (последовательно, как RNN). Другая кодирует ее как параллельную редукцию (broadcast, как attention). Та же математика, другой граф зависимостей.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Мы замеряем время для обоих вариантов на последовательностях до 100,000 элементов. RNN-версия имеет O(N) и один CPU pipeline. Даже в чистом Python редукция в стиле attention обгоняет ее на длине ≥ 1,000, потому что Python-овский `sum()` реализован на C и итерируется без overhead интерпретатора на каждом шаге.

### Step 2: посчитайте теоретические операции

Оба алгоритма делают N сложений. Разница — в *глубине зависимостей*: сколько операций должно произойти последовательно, прежде чем следующая сможет начаться. Глубина RNN = N. Глубина attention = log(N) при tree reduction или 1 при parallel scan. GPU-время определяет глубина, а не число операций.

### Step 3: эмпирическое масштабирование на длинных последовательностях

Мы печатаем таблицу timings, которая делает разрыв O(N) видимым. На ноутбуке Mac 2026 года последовательности короче 1,000 элементов слишком быстры для измерения. Последовательности из 100,000 показывают чистый линейный scan. Масштабируйте это до 16,384-token transformer с эквивалентом 12-layer LSTM, и станет понятно, почему wall-clock обучения был блокером в 2016 году.

## Use It

Когда в 2026 году все еще выбирать RNN:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

State-space models (SSMs), такие как Mamba, по сути являются RNN со структурированной параметризацией, которая дает им лучшее из обоих миров: `O(N)` память scan и параллельное обучение через selective scan. Они восстанавливают 90% качества transformer при лучшем масштабировании длинного контекста. В 2026 году большинство frontier labs обучают гибридные SSM+transformer модели (например, Jamba, Samba) — recurrence не умерла, она стала компонентом.

## Ship It

См. `outputs/skill-architecture-picker.md`. Skill выбирает архитектуру для новой задачи с последовательностями по ограничениям длины, throughput и training budget. Он всегда должен отказываться рекомендовать чистую RNN для обучающих запусков выше 1B токенов без явного указания trade-off.

## Упражнения

1. **Easy.** Возьмите `rnn_style` из `code/main.py` и замените скалярное скрытое состояние на вектор скрытых состояний длины 64. Измерьте заново. Насколько последовательный overhead растет с размерностью hidden state?
2. **Medium.** Реализуйте parallel prefix-sum (Hillis-Steele scan) на чистом Python. Проверьте, что он дает тот же численный результат, что и serial scan на длине 1024. Посчитайте глубину.
3. **Hard.** Перенесите редукцию в стиле attention в PyTorch на GPU. Замерьте оба варианта, перебирая длину последовательности от 64 до 65,536. Постройте график и объясните форму кривой.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Вычисление, где шаг `t` зависит от шага `t-1`, что вынуждает последовательное выполнение вдоль временной оси. |
| Serial depth | "How deep the graph is" | Самая длинная цепочка зависимых операций; ограничивает wall-clock даже на бесконечном железе. |
| Attention | "Let tokens look at each other" | Взвешенная сумма `sum_j a_ij v_j`, где `a_ij` получается из score сходства между позициями i и j. |
| Context window | "How much the model sees" | Число позиций, которое attention layer может принять на вход; квадратичная стоимость памяти масштабируется здесь. |
| Inductive bias | "Assumptions baked into the architecture" | Априорное предположение о виде данных; CNN предполагают translation invariance, RNN предполагают recency. |
| State-space model | "RNN with algebra behind it" | Recurrence, параметризованная для параллельного обучения через структурированные state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Память attention = `O(N²)` по длине последовательности; Flash Attention скрывает константы, но не scaling. |

## Дополнительное чтение

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — статья, которая убрала recurrence из mainstream NLP.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — где attention родился, будучи прикрученным к RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — оригинальная статья LSTM, для истории.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — современный рекуррентный ответ transformers.
