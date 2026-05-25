# DualPipe Parallelism

> DeepSeek-V3 обучали на 2,048 H800 GPUs с MoE experts, разбросанными по nodes. Cross-node expert all-to-all communication стоила 1 GPU-hour связи на каждый 1 GPU-hour compute. GPUs простаивали половину времени. DualPipe (DeepSeek, Dec 2024) — bidirectional pipeline, который перекрывает forward и backward computation с all-to-all comms, которые они запускают. Bubbles уменьшаются, throughput растет, а хранение двух copies model parameters (то самое "dual" в названии) дешево, когда Expert Parallelism уже распределяет experts по ranks. Этот урок — Learn-type walkthrough того, что реально делает DualPipe и почему refinement DualPipeV от Sea AI Lab убирает 2x parameter cost ценой немного более плотного bubble.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## Цели обучения

- Назвать четыре components DualPipe forward-backward chunk и почему каждая получает собственное overlap window.
- Объяснить pipeline bubble problem в масштабе и что "bubble-free" означает на практике, а не в маркетинге.
- Вручную пройти DualPipe schedule для 8 PP ranks и 16 micro-batches и подтвердить, что forward и reverse streams заполняют idle slots друг друга.
- Сформулировать tradeoff DualPipeV (Sea AI Lab, 2025): убирает 2x parameter replication ценой чуть большего bubble, когда Expert Parallelism неактивен.

## Проблема

Training 671B MoE model на 2k H800 GPUs упирается в три взаимно усиливающихся bottlenecks:

1. **Memory pressure.** Каждый GPU держит slice модели. Activation memory при sequence 8k, 61 layers и 128 heads огромна.
2. **Pipeline bubbles.** Traditional pipeline parallelism (GPipe, 1F1B) оставляет GPUs idle, пока они ждут input своего stage или gradient. При 8 stages около 12% GPU time может быть bubble даже с 1F1B scheduling.
3. **Cross-node all-to-all.** MoE с expert parallelism разбрасывает experts по nodes. Каждый forward pass запускает all-to-all для dispatch tokens к experts и еще один для combine. На 2k GPUs это легко превращается в ratio compute-to-comm 1:1.

У каждой проблемы есть отдельное решение: gradient checkpointing для memory, Zero Bubble (Sea AI Lab, 2023) для pipeline bubbles, expert-parallel comm kernels для all-to-all. DualPipe заставляет их работать вместе. Schedule перекрывает compute и comm внутри одного forward-backward chunk, вводит micro-batches одновременно с обоих концов pipeline и использует получившийся schedule, чтобы спрятать all-to-all внутри compute windows.

Reported result: почти устранение pipeline bubbles, более 95% GPU utilization в 14.8T-token training run DeepSeek-V3.

## Концепция

### Pipeline parallelism refresher

Разделите N-layer model по P devices. Device `i` держит layers `i * N/P .. (i+1) * N/P - 1`. Micro-batch проходит forward через devices 0 to P-1, затем backward от P-1 к 0. Каждый device может начать свой forward stage только когда previous device отправит output, и может начать backward только когда downstream device отправит upstream gradient.

GPipe (Huang et al., 2019) schedules один micro-batch за раз, что тратит большую часть GPU time впустую. 1F1B (Narayanan et al., 2021) interleaves forward and backward passes для нескольких micro-batches. Zero Bubble (Qi et al., 2023) делит backward pass на две части — backward-for-input (B) и backward-for-weights (W) — и schedules их, чтобы заполнить bubble. После Zero Bubble pipeline почти tight.

DualPipe — следующий шаг. Он добавляет две идеи:

### Idea 1: chunk decomposition

Каждый forward chunk делится на четыре components:

- **Attention.** Q/K/V projections, attention, output projection.
- **All-to-all dispatch.** Cross-node communication, отправляющая tokens к их experts.
- **MLP.** MoE expert computation.
- **All-to-all combine.** Cross-node communication, возвращающая expert outputs.

Backward chunk добавляет gradient versions каждого компонента. DualPipe schedules их так, что all-to-all dispatch идет параллельно attention compute следующего chunk, а all-to-all combine — параллельно MLP compute следующего chunk.

### Idea 2: bidirectional scheduling

Большинство pipeline schedules вводит micro-batches со stage 0 и гонит к stage P-1. DualPipe вводит micro-batches с ОБОИХ концов. Stage 0 видит forward micro-batches, начинающиеся там; stage P-1 тоже видит forward micro-batches, начинающиеся там. Два streams встречаются в середине.

Чтобы это работало, device `i` должен держать И early-pipeline layer `i`, И late-pipeline layer `P - 1 - i`. Это "dual" часть DualPipe: каждый device хранит две copies layers, нужных для обслуживания (по одной для каждого direction). В масштабе DeepSeek-V3 это 2x parameter replication cost. Он доступен, потому что Expert Parallelism уже так тонко распределяет MoE experts, что двойная репликация non-expert layers — мелочь.

Критично, что forward stream в одном direction и backward stream в другом direction перекрываются ровно там, где были бы bubbles в single-direction schedule. Bubbles исчезают.

### A hand-traced schedule

Рассмотрим P = 4 ranks, 8 micro-batches, разделенные на 4 forward / 4 reverse. Время идет слева направо; строки — device ranks.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Обозначение "F4/F5R" читается так: rank 1 выполняет forward micro-batch 4 (идет left-to-right по pipeline) И forward micro-batch 5 (идет right-to-left) в одном time slot. Именно это operationally означает "bidirectional".

На rank 2 cross streams перекрываются раньше, на rank 0 и P-1 — позже. В stable middle phase schedule каждый rank выполняет forward-of-X-direction, перекрытый с backward-of-Y-direction. Compute занят. All-to-all dispatches для forward pass прячутся внутри backward compute. All-to-all combines прячутся внутри forward compute. Bubbles выжимаются.

### Bubble accounting

Standard 1F1B pipeline bubble (wasted time per rank):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble refinement снижает это, но не до нуля. DualPipe в stable phase имеет zero bubble, если micro-batch count делится на 2 * pipeline depth. За пределами stable phase (warmup and cooldown) bubble остается, но не растет с number of micro-batches — ключевое свойство, которое подчеркивает paper.

В маркетинговых терминах: "bubble-free". В технических: bubbles не растут с micro-batch count. Follow-up анализ Sea AI Lab (DualPipeV / Cut-in-half) показывает, что полный zero-bubble получается только когда Expert Parallelism не является bottleneck; с EP-driven all-to-all некоторый scheduling compromise всегда присутствует.

### DualPipeV — the refinement

Sea AI Lab (2025) заметила, что 2x parameter replication расточительна, когда EP comm overlap не является целью. Их schedule DualPipeV сворачивает bidirectional injection в "V-shape" schedule, работающий с одной parameter copy. Bubble чуть больше, чем у DualPipe, но memory savings существенные. DeepSeek принял DualPipeV в open-source DualPipe implementation как EP-off mode.

Tradeoff:

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### What it means for a 14.8T-token run

Pre-training DeepSeek-V3 потребил 14.8T tokens на 2,048 H800 GPUs примерно за 2.8M GPU-hours. С naive 1F1B они потеряли бы 12-15% на pipeline bubbles — 340-420K GPU-hours, достаточно для обучения full 70B model. DualPipe вернул большую часть этого. Точно количественно отделить вклад сложно без internal logs, но в paper claim — более 95% GPU utilization в среднем по training.

Для smaller runs (меньше 1k GPUs) DualPipe избыточен — pipeline bubbles меньше относительно total cost, а dense-model training редко упирается в all-to-all bottleneck. Для frontier MoE training в multi-thousand GPU scale он фактически обязателен.

### Where it sits in the stack

- Complementary to **FSDP** (Phase 10 · 05). FSDP shards model parameters across ranks; DualPipe schedules compute across ranks. Они сочетаются.
- Compatible with **ZeRO-3** gradient sharding. Bookkeeping для two-copy replication должен взаимодействовать с sharded gradients ZeRO.
- Requires **custom all-to-all kernels**, tuned под конкретную cluster topology. Open-source kernels DeepSeek — reference implementation.

## Использование

`code/main.py` — pipeline schedule simulator. Он принимает `(P, n_micro_batches, schedule)` и печатает stable-phase utilization для 1F1B, Zero Bubble, DualPipe и DualPipeV. Это teaching tool — числа совпадают с qualitative claims в papers, но не являются claim о production measured speedup.

Ценность simulator: запускайте его с разными P и micro-batch counts и наблюдайте, как bubble fraction растет для 1F1B, но не для DualPipe.

Integration considerations для real training run:

- Выберите pipeline-parallel depth, который чисто делит micro-batch count.
- Убедитесь, что expert-parallel mesh поддерживает bidirectional all-to-all. Kernels DeepSeek — reference.
- Заложите неделю debugging time на сам schedule в первый раз. Bookkeeping fiddly.
- Monitor GPU utilization per rank, не только aggregate. Выигрыш DualPipe приходит от tightening stragglers.

## Результат

Этот урок создает `outputs/skill-dualpipe-planner.md`. По training cluster specification (GPU count, topology, interconnect, model shape) он рекомендует pipeline parallelism strategy, scheduling algorithm и expected bubble fraction в целевом масштабе.

## Упражнения

1. Запустите `code/main.py` на `(P=8, micro_batches=16, schedule=dualpipe)` и `(P=8, micro_batches=16, schedule=1f1b)`. Посчитайте GPU utilization difference и выразите его как recovered GPU-hours per million tokens of training.

2. Вручную набросайте schedule table для `(P=4, micro_batches=8, schedule=dualpipe)`. Отметьте каждый time slot micro-batch ID и direction. Найдите первый time slot, где bubbles отсутствуют.

3. Прочитайте Figure 5 технического отчета DeepSeek-V3 (arXiv:2412.19437). Найдите overlap window для all-to-all dispatch внутри DualPipe forward chunk. Объясните, как compute schedule прячет его.

4. Посчитайте 2x parameter overhead DualPipe для 70B dense model с P=8 pipeline stages и 671B MoE model с P=16 pipeline stages. Покажите, почему overhead в MoE case пропорционально меньше (большинство parameters — experts, sharded across large EP group).

5. Сравните DualPipe с Chimera (конкурирующим bidirectional scheduler 2021 года). Назовите два конкретных свойства, которые DualPipe добавил, а Chimera не имела, используя Section 3.4 paper как reference.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Pipeline bubble | "Idle time per rank" | GPU cycles, потраченные впустую, потому что pipeline stage ждет input или gradient |
| 1F1B | "Default pipeline schedule" | One forward / one backward interleaved scheduling; baseline, который обгоняет DualPipe |
| Zero Bubble | "Sea AI Lab 2023" | Делит backward на B (input gradient) и W (weight gradient); почти полностью tightening pipeline |
| DualPipe | "DeepSeek-V3 schedule" | Bidirectional pipeline + compute-comm overlap; bubbles не растут с micro-batch count |
| DualPipeV | "Cut-in-half" | V-shape refinement, который убирает 2x parameter replication ценой чуть больших bubbles |
| Chunk | "Unit of pipeline work" | Forward или backward pass одного micro-batch через один pipeline stage |
| All-to-all dispatch | "Send tokens to experts" | Cross-node comm, маршрутизирующая tokens к назначенным MoE experts |
| All-to-all combine | "Bring expert outputs back" | Cross-node comm, собирающая expert outputs после MLP |
| Expert Parallelism (EP) | "Experts across GPUs" | Shards MoE experts across ranks, так что разные GPUs держат разных experts |
| Pipeline Parallelism (PP) | "Layers across GPUs" | Shards model layers across ranks; dimension, которую schedules DualPipe |
| Bubble fraction | "Wasted GPU time" | (bubble_time / total_time); доля, которую DualPipe стремится к нулю |

## Дополнительное чтение

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — основной reference по DualPipe
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — open-source reference implementation, включая режим DualPipeV (Cut-in-half)
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — predecessor Zero Bubble
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — анализ DualPipeV, повлиявший на EP-off mode DeepSeek
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — schedule 1F1B, с которым сравнивается DualPipe
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — исходная статья о pipeline parallelism и bubble problem
