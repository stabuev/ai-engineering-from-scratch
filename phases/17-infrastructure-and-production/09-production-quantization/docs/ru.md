# Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Формат quantization — не универсальный выбор: он является функцией hardware, serving engine и workload. GGUF Q4_K_M или Q5_K_M доминирует на CPU и edge, поставляется через llama.cpp и Ollama. GPTQ выигрывает внутри vLLM, когда нужен multi-LoRA на одной base model. AWQ с kernels Marlin-AWQ дает ~741 tok/s на модели класса 7B с лучшим Pass@1 при INT4 — default 2026 для datacenter production. FP8 остается серединой на Hopper, Ada и Blackwell — почти без потерь качества и широко поддерживается. NVFP4 и MXFP4 (Blackwell microscaling) агрессивны и требуют per-block validation. Две ловушки кусают команды: calibration dataset должен совпадать с deployment domain, а KV cache отделен от weight quantization — урок AWQ "my model is 4 GB now" забывает про 10-30 GB KV cache при production batch sizes.

**Тип:** Learn
**Языки:** Python (stdlib, toy memory and throughput comparison across formats)
**Предварительные требования:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Время:** ~75 minutes

## Цели обучения

- Назвать шесть production quantization formats и их sweet spots в 2026.
- Выбрать формат по hardware (CPU vs GPU, Hopper vs Blackwell), engine (vLLM, TRT-LLM, llama.cpp) и workload (routine chat, reasoning, multi-LoRA).
- Вычислить сэкономленную память weights и нетронутый KV cache для выбранного формата.
- Назвать ловушку calibration dataset, которая ухудшает quantized models на domain traffic.

## Проблема

Quantization уменьшает memory и HBM bandwidth, что именно нужно decode. FP16 70B model — это 140 GB weights. Quantize weights до INT4 (AWQ или GPTQ), и модель становится 35 GB — помещается в один H100 с запасом для KV cache, что важно, потому что при 128 concurrent sequences с 2k context один KV cache занимает 20-30 GB.

Но quantization не бесплатна. Агрессивная quantization ухудшает качество, особенно на reasoning-heavy tasks. Разные форматы работают с разными engines. Разное hardware нативно поддерживает разные precisions. Зоопарк форматов 2026 реален, и нельзя копировать чужой выбор — нужно выбирать под свой stack.

## Концепция

### Шесть форматов

| Format | Bits | Sweet spot | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptops | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA on vLLM | vLLM, TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### GGUF — default для CPU/edge

GGUF — file format, а не схема quantization сама по себе: он упаковывает K-quant variants (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) в один container. Q4_K_M и Q5_K_M — production defaults, качество близко к BF16 при 4-5 bits. Лучший выбор для CPU или edge serving, потому что llama.cpp с большим отрывом самый быстрый CPU inference engine.

Штраф throughput в vLLM: ~93 tok/s на 7B — формат не оптимизирован под GPU kernels. Используйте GGUF, когда deployment target — CPU/edge. В остальных случаях нет.

### GPTQ — multi-LoRA во vLLM

GPTQ — post-training quantization algorithm с calibration pass. Marlin kernels делают его быстрым на GPU (2.6x speedup vs non-Marlin GPTQ). ~712 tok/s на 7B.

Уникальная победа: GPTQ-Int4 поддерживает LoRA adapters во vLLM. Если вы serving base model плюс 10-50 fine-tuned variants (каждый как LoRA), GPTQ — ваш путь. NVFP4 пока не поддерживает LoRA по состоянию на early 2026.

### AWQ — default для datacenter GPU

Activation-aware Weight Quantization. Защищает ~1% наиболее salient weights во время quantization. Marlin-AWQ kernels: 10.9x speedup vs naive. ~741 tok/s на 7B, лучший Pass@1 среди INT4 formats.

Выбирайте AWQ для нового GPU serving, если вам не нужен multi-LoRA (GPTQ) или агрессивный Blackwell FP4 (NVFP4).

### FP8 — надежная середина

8-bit floating point. Почти без потерь. Широкая поддержка. Hopper Tensor Cores нативно ускоряют FP8. Blackwell наследует это. FP8 — safe default 2026, когда качество не обсуждается (reasoning, medical, code-gen). Экономия памяти вдвое меньше, чем у INT4, но риск для качества намного ниже.

### MXFP4 / NVFP4 — агрессивный Blackwell

Microscaling FP4. У каждого block of weights есть собственный scale factor. Агрессивно, но аппаратно ускорено на Blackwell Tensor Cores. Вдвое меньше bytes per token по сравнению с FP8 — экономический выигрыш из Phase 17 · 07.

Ограничения:
- Пока нет поддержки LoRA (early 2026).
- Quality drop заметен на reasoning-heavy workloads.
- Валидируйте на своем eval set для каждой model.

### Ловушка calibration

AWQ и GPTQ требуют calibration dataset — обычно C4 или WikiText. Для domain models (code, medical, legal) calibration на generic web text позволяет алгоритму принять неверные решения о том, какие weights защищать. Pass@1 на HumanEval может упасть на несколько points.

Исправление: calibrate на in-domain data. Обычно достаточно сотен domain samples. Перед shipping тестируйте на eval set.

### Ловушка KV cache

AWQ сжимает weights до 4 bits. KV cache отделен и остается FP16/FP8. Для 70B model с AWQ:

- Weights: ~35 GB (INT4 from 140 GB).
- KV cache at 128 concurrent × 2k context: ~20 GB.
- Activations: ~5 GB.
- Total: ~60 GB — fits on H100 80GB.

Наивное "I quantized my model to 4 GB" забывает остальные 30-50 GB. Планируйте HBM целиком.

Отдельно, KV cache quantization (FP8 KV или INT8 KV) — другой выбор со своими tradeoffs: он напрямую влияет на attention accuracy и не является бесплатной победой.

### AWQ INT4 опасен для reasoning

Chain-of-thought, math, code-gen с long context — они заметно страдают от aggressive quantization. AWQ INT4 теряет ~3-5 points на MATH. Для reasoning-heavy workloads отправляйте FP8 или BF16; принимайте memory cost.

### Руководство выбора 2026

- CPU/edge serve: GGUF Q4_K_M. Done.
- GPU serve, routine chat, no LoRA: AWQ.
- GPU serve, multi-LoRA: GPTQ with Marlin.
- Reasoning workload: FP8.
- Blackwell datacenter, validated quality: NVFP4 + FP8 KV.
- Ambiguous: run a 1,000-sample eval on each candidate format.

## Используйте это

`code/main.py` вычисляет memory footprint (weights + KV + activations) и относительный throughput по шести форматам для диапазона model sizes. Показывает, где доминирует KV cache, где окупается weight compression и где FP8 — safe pick.

## Отправьте в прод

Этот урок создает `outputs/skill-quantization-picker.md`. По hardware, model size, workload type и quality tolerance он выбирает формат и создает calibration/validation plan.

## Упражнения

1. Запустите `code/main.py`. Для 70B model при 128 concurrent с 2k context вычислите total HBM для каждого формата. Какой формат позволяет поместиться на один H100 80GB?
2. У вас 7B coding model. Выберите формат и обоснуйте. Если вы ошиблись с quality tolerance, каков recovery path?
3. Вычислите размер calibration dataset, нужный для calibration AWQ для medical domain model. Почему больше данных не всегда лучше?
4. Прочитайте paper или release notes Marlin-AWQ kernel. Объясните в трех предложениях, почему AWQ достигает 741 tok/s на 7B, а raw GPTQ — ~712.
5. Когда имеет смысл сочетать AWQ weights с FP8 KV cache, а когда оставить KV в BF16?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| GGUF | "llama.cpp format" | File format bundling K-quant variants; CPU/edge default |
| Q4_K_M | "Q4 K M" | 4-bit K-quant medium; the production GGUF default |
| GPTQ | "gee pee tee q" | Post-train INT4 with calibration; supports LoRA in vLLM |
| AWQ | "a w q" | Activation-aware INT4; Marlin kernels; best Pass@1 at INT4 |
| Marlin kernels | "fast INT4 kernels" | Custom CUDA kernels for INT4 on Hopper; 10x speedup |
| FP8 | "eight-bit float" | Safe precision default on Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "microscaling four" | Blackwell 4-bit FP with per-block scale factors |
| Calibration dataset | "cal data" | Input text used to pick quantization parameters; must match domain |
| KV cache quantization | "KV INT8" | Separate choice from weights; affects attention accuracy |

## Дополнительное чтение

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — сравнительные benchmarks.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — throughput numbers by format.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — выбор по форматам.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — поддерживаемые formats and flags.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — исходная формулировка AWQ.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — исходная формулировка GPTQ.
