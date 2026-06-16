# Capstone 07 — End-to-End Fine-Tuning Pipeline (Data to SFT to DPO to Serve)

> 8B model, обученная на ваших данных, DPO-aligned на ваших предпочтениях, quantized, speculative-decoded и обслуживаемая с измеримым $/1M tokens. Open stack 2026 года — Axolotl v0.8, TRL 0.15, Unsloth для итераций, GPTQ/AWQ/GGUF для quantization, vLLM 0.7 with EAGLE-3 для serving. Capstone — воспроизводимо запустить весь pipeline: YAML на входе, served endpoint на выходе, и опубликовать model card по 2026 Model Openness Framework.

**Тип:** Capstone
**Языки:** Python (pipeline), YAML (configs), Bash (scripts)
**Предварительные требования:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Задействованные фазы:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Время:** 35 hours

## Цели обучения

- Построить сквозной пайплайн: дообучить 8B-модель на своих данных, выровнять DPO, квантовать и обслуживать.
- Измерять стоимость в $/1M токенов и латентность.
- Добавить speculative decoding в путь обслуживания.

## Проблема

В 2026 году у каждой серьезной AI-команды под рукой есть fine-tuning pipeline. Не потому, что они выпускают frontier base model, а потому что downstream adaptation — domain SFT, DPO against labeled preferences, distilled drafts for speculative decoding, serving with EAGLE-3 — дает измеримые выигрыши. Axolotl v0.8 ведет multi-GPU SFT configs. TRL 0.15 ведет DPO and GRPO. Unsloth дает быстрые single-GPU iteration. vLLM 0.7 with EAGLE-3 поднимает decode throughput в 2-3x без потери качества. Инструменты работают; мастерство — в YAMLs, data hygiene и eval discipline.

Вы проведете 8B base (Llama 3.3, Qwen3 или Gemma 3) через SFT, затем DPO на task-specific data, quantize for serving и измерите gains против lm-evaluation-harness, RewardBench-2, MT-Bench-v2 и MMLU-Pro. Вы подготовите model card по 2026 Model Openness Framework. Смысл — воспроизводимость: одна команда перезапускает весь pipeline end to end.

## Концепция

Pipeline состоит из пяти стадий. **Data**: dedup (MinHash / Datatrove), quality filter (Nemotron-CC style classifier), PII scrub, split-hygiene check against public benchmark contamination. **SFT**: Axolotl YAML, ZeRO-3 on 8xH100, cosine schedule, packed sequences, 2-3 epochs. **DPO or GRPO**: TRL config, 1 epoch, preference pairs either human-labeled or model-judged, beta tuning. **Quantize**: GPTQ + AWQ + GGUF для deployment flexibility. **Serve**: vLLM 0.7 with EAGLE-3 speculative heads или SGLang with SpecForge, K8s deployment, HPA on queue-wait.

Ablations — deliverable: SFT-only vs SFT+DPO vs SFT+GRPO на трех task-specific benchmarks. Serving metrics: tokens/s at batch 1 / 8 / 32, EAGLE-3 acceptance rate, $/1M tokens. Safety eval: Llama Guard 4 pass rate. Model card: bias evaluations, reproducibility seeds, data licensing.

## Архитектура

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Стек

- Data: Datatrove for dedup, Nemotron-CC classifier for quality, Presidio for PII
- Base: Llama 3.3 8B, Qwen3 14B, or Gemma 3 12B
- SFT: Axolotl v0.8 with ZeRO-3, Flash Attention 3, packed sequences
- Preference tuning: TRL 0.15 for DPO or GRPO; Unsloth for single-GPU iteration
- Quantization: GPTQ (Marlin), AWQ, GGUF via llama.cpp
- Serving: vLLM 0.7 with EAGLE-3 speculative decoding (or SGLang 0.4 + SpecForge)
- Eval: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Safety eval: Llama Guard 4, ShieldGemma-2
- Infrastructure: Kubernetes + NVIDIA device plugin, HPA on queue-wait metric
- Observability: W&B for training, Langfuse for inference

## Постройте это

1. **Data pipeline.** Запустите Datatrove dedup на raw corpus. Примените Nemotron-CC-style quality classifier. Presidio очищает PII. Запишите train/val splits с explicit seed.

2. **Contamination check.** Для каждого validation split вычислите MinHash against MMLU-Pro, MT-Bench-v2, RewardBench-2 test sets. Отклоняйте любой overlap.

3. **Axolotl SFT.** YAML with ZeRO-3, FA3, sequence packing. 2-3 epochs on 8xH100. Логируйте в W&B.

4. **TRL DPO / GRPO.** Возьмите SFT checkpoint, запустите one epoch of DPO on preference pairs или GRPO with a verifiable reward on math/code. Сделайте sweep beta.

5. **Quantize.** Выпустите три quants: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M for llama.cpp. Запишите size and nominal throughput.

6. **Serve with speculative decoding.** vLLM 0.7 config with EAGLE-3 draft heads trained via Red Hat Speculators. Измерьте acceptance rate and tail latency at batch 1 / 8 / 32. Сообщите $/1M tokens vs Anthropic / OpenAI on the same eval.

7. **Eval matrix.** Запустите lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro на base, SFT-only, SFT+DPO, SFT+GRPO. Подготовьте table.

8. **Safety eval.** Llama Guard 4 pass rate on the dev set. ShieldGemma-2 output filter.

9. **Model card.** MOF 2026 template: data, training, eval, safety, license, reproducibility section with YAMLs and commit SHAs.

## Используйте это

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Отгрузите это

`outputs/skill-finetuning-pipeline.md` описывает deliverable. Одна команда прогоняет data through SFT through DPO through quant through serve through eval и выдает model card + served endpoint.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Eval delta vs base | Measured gain on target tasks (MMLU-Pro, MT-Bench-v2, task-specific) |
| 20 | Воспроизводимость pipeline | One command reruns end to end with identical seeds |
| 20 | Data hygiene | Dedup rate, PII scrub coverage, contamination check green |
| 20 | Serving efficiency | tokens/s at bs=1/8/32, EAGLE-3 acceptance rate, $/1M tokens |
| 15 | Model card + safety eval | 2026 MOF completeness + Llama Guard 4 pass rate |
| **100** | | |

## Упражнения

1. Запустите SFT-only vs SFT+DPO vs SFT+GRPO на одном task-specific benchmark. Сообщите, какой preference method выигрывает и насколько.

2. Замените Llama 3.3 8B на Qwen3 14B. Измерьте $/1M tokens при matched quality.

3. Измерьте EAGLE-3 acceptance rate на domain data vs generic ShareGPT. Сообщите delta и что она означает для latency budgets.

4. Внесите 1% contamination: слейте MMLU-Pro answers в training data, затем перезапустите eval. Посмотрите, как MMLU-Pro accuracy нереалистично подскакивает. Постройте contamination-check CI gate, который это ловит.

5. Добавьте LoRA SFT как альтернативу full fine-tune. Измерьте quality gap при 10x lower memory.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Axolotl | "SFT trainer" | Unified YAML-driven trainer for SFT, DPO, and distillation |
| TRL | "Preference tuner" | Hugging Face library for DPO, GRPO, PPO on LLMs |
| GRPO | "Group-relative policy optimization" | DeepSeek R1's RL recipe with verifiable rewards |
| EAGLE-3 | "Speculative decoding draft" | Draft heads that predict N tokens ahead; vLLM verifies with target model |
| MOF | "Model Openness Framework" | 2026 standard for grading model releases on data, code, license |
| Contamination check | "Split hygiene" | MinHash-based detection of test-set leakage into training |
| Acceptance rate | "EAGLE / MTP metric" | Fraction of drafted tokens the target model accepts |

## Дополнительное чтение

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — эталонный SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO and GRPO reference implementations
- [Unsloth](https://github.com/unslothai/unsloth) — single-GPU iteration reference
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO methodology
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — эталонный serving stack
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — alternate speculative-decoding trainer
- [Model Openness Framework 2026](https://isocpp.org/) — open-release grading standard
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — canonical eval runner
