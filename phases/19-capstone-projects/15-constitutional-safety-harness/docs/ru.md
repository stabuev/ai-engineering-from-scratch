# Capstone 15 — Constitutional Safety Harness + Red-Team Range

> Anthropic Constitutional Classifiers, Meta Llama Guard 4, Google ShieldGemma-2, NVIDIA Nemotron 3 Content Safety и X-Guard для multilingual coverage определили стек safety-classifier 2026 года. garak, PyRIT, NVIDIA Aegis и promptfoo стали стандартными adversarial evaluation tools. NeMo Guardrails v0.12 связывает их в production pipeline. Этот capstone соединяет все вместе: layered safety harness вокруг target app, autonomous red-team agent с 6+ attack families и constitutional self-critique run, который дает измеримый harmlessness delta.

**Тип:** Capstone
**Языки:** Python (safety pipeline, red team), YAML (policy configs)
**Пререквизиты:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Отрабатываемые фазы:** P10 · P11 · P13 · P14 · P18
**Время:** 25 часов

## Цели обучения

- Построить constitutional safety harness плюс red-team полигон.
- Сочетать классификатор входа/выхода (в стиле Llama Guard) с конституционной самокритикой.
- Оценивать harness против red-team атак.

## Задача

Передний край LLM safety в 2026 году — не в том, работают ли classifiers (примерно работают), а в том, как правильно собрать их вокруг production app без чрезмерных отказов и очевидных дыр. Llama Guard 4 обрабатывает English policy violations. X-Guard (132 languages) обрабатывает multilingual jailbreak. ShieldGemma-2 ловит image-based prompt injection. NVIDIA Nemotron 3 Content Safety покрывает enterprise categories. Anthropic Constitutional Classifiers — отдельный подход, используемый во время training, а не serving.

Эволюция attacks тоже важна. PAIR и TAP автоматизируют jailbreak discovery. GCG запускает gradient-based suffix attacks. Multi-turn и code-switch attacks используют agent memory. Любой deployed LLM нуждается в red-team range — garak и PyRIT являются canonical drivers — плюс документированные mitigations и findings с CVSS scoring.

Ты укрепишь target application (либо 8B instruction-tuned model, либо один из RAG chatbots из других capstones), запустишь против него 6+ attack families и получишь before/after harmlessness measurement.

## Концепция

Safety pipeline состоит из пяти слоев. **Input sanitize**: удалить zero-width chars, декодировать base64/rot13, нормализовать Unicode. **Policy layer**: NeMo Guardrails v0.12 rails (off-domain, toxicity, PII extraction). **Classifier gate**: Llama Guard 4 на input, X-Guard на non-English, ShieldGemma-2 на image inputs. **Model**: target LLM. **Output filter**: Llama Guard 4 на output, Presidio PII scrub, citation enforcement где применимо. **HITL tier**: outputs с high-risk flags уходят в Slack queue.

Red-team range запускается по scheduler. PAIR и TAP автономно находят jailbreaks. GCG запускает gradient-based suffix attacks. ASCII / base64 / rot13 encoding attacks. Multi-turn attacks (persona adoption, memory exploitation). Code-switch attacks (смешивание English со Swahili или Thai). Каждый run создает structured findings file с CVSS scoring и disclosure timeline.

Constitutional-self-critique run — это training-time intervention. Возьми 1k harmful-attempt prompts, дай model составить response, раскритикуй его по written constitution (do-not-harm rules) и дообучи на critique loop. Измерь before/after harmlessness delta на held-out eval.

## Архитектура

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Стек

- Safety classifiers: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- Guardrail framework: NeMo Guardrails v0.12 + OPA
- Red-team drivers: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- Jailbreak agents: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG suffix
- Constitutional training: Anthropic-style self-critique loop + SFT on critiques
- PII scrub: Presidio
- Target: 8B instruction-tuned model или один из RAG chatbots из других capstones

## Сборка

1. **Настройка target.** Подними 8B instruction-tuned model на vLLM (или переиспользуй RAG chatbot из другого capstone). Это app under test.

2. **Обертка safety pipeline.** Оберни target пятислойным pipeline. Проверь, что каждый слой отдельно наблюдаем (span per layer in Langfuse).

3. **Покрытие classifiers.** Загрузи Llama Guard 4, X-Guard (multilingual), ShieldGemma-2 (image). Запусти каждый на small labeled set, чтобы установить baselines.

4. **Red-team scheduler.** Запланируй garak, PyRIT, PAIR agent, TAP agent, GCG runner, multi-turn attacker и code-switch attacker. Каждый запускается в отдельной queue.

5. **Attack suite.** Шесть attack families: (1) PAIR automated jailbreak, (2) TAP tree-of-attacks, (3) GCG gradient suffix, (4) ASCII / base64 / rot13 encoding, (5) multi-turn persona, (6) multilingual code-switch. Отчитай success rate по каждой family.

6. **Constitutional self-critique.** Подготовь 1k harmful-attempt prompts. Для каждого target составляет response. Critic LLM оценивает по written constitution ("do no harm," "cite evidence," "refuse illegal requests"). Prompts, против которых critic возражает, переписываются; target fine-tunes на critique-improved pairs. Измерь before/after harmlessness на held-out eval.

7. **Измерение over-refusal.** Отслеживай false-positive rate на benign prompt suite (например, XSTest). Target должен оставаться helpful на benign questions.

8. **CVSS scoring.** Для каждого successful jailbreak поставь score по CVSS 4.0 (attack vector, complexity, impact). Подготовь disclosure timeline и mitigation plan.

9. **Range automation.** Все вышеописанное запускается по cron; findings пишутся в queue; over-refusal regression alerts отправляются в Slack.

## Использование

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Что сдать

`outputs/skill-safety-harness.md` — deliverable. Production-grade layered safety pipeline плюс воспроизводимый red-team range с before/after harmlessness deltas.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Покрытие attack-surface | 6+ attack families exercised, 2+ languages |
| 20 | Компромисс true-positive / false-positive | Attack block rate vs XSTest benign pass rate |
| 20 | Self-critique delta | Before/after harmlessness на held-out eval |
| 20 | Документация и disclosure | Findings с CVSS scoring и timeline |
| 15 | Automation and repeatability | Все запускается по cron с alerts |
| **100** | | |

## Упражнения

1. Запусти garak plugin для prompt-injection на RAG chatbot и сравни attack success rate с output-filter layer и без него.

2. Добавь седьмую attack family: indirect prompt injection через retrieved documents. Измерь дополнительную защиту, которая нужна.

3. Реализуй режим "refuse-with-help": когда guardrail блокирует запрос, target предлагает более безопасный связанный answer вместо flat refusal. Измерь XSTest delta.

4. Пробел в multilingual coverage: найди language, где X-Guard работает хуже. Предложи fine-tune dataset под него.

5. Запусти constitutional self-critique на 30B model и измерь, масштабируется ли delta.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|------------------------|
| Layered safety | "Defense in depth" | Несколько guardrails на input, gate, output, HITL |
| Llama Guard 4 | "Meta's safety classifier" | Эталонный input/output content classifier 2026 года |
| PAIR | "Jailbreak agent" | Paper (Chao et al.) про LLM-driven jailbreak discovery |
| TAP | "Tree-of-Attacks" | Tree-search вариант PAIR |
| GCG | "Greedy coordinate gradient" | Gradient-based adversarial suffix attack |
| Constitutional self-critique | "Обучение в стиле Anthropic" | Target drafts -> critic scores -> rewrite -> retrain |
| XSTest | "Benign probe set" | Benchmark для over-refusal regression |
| CVSS 4.0 | "Severity score" | Стандартная оценка vulnerability severity для safety findings |

## Дополнительное чтение

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — reference для training-time
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — input/output classifier 2026 года
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — image + multimodal safety
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — enterprise reference
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — multilingual safety для 132 languages
- [garak](https://github.com/NVIDIA/garak) — red-team toolkit от NVIDIA
- [PyRIT](https://github.com/Azure/PyRIT) — red-team framework от Microsoft
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail framework
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — paper про jailbreak agent
