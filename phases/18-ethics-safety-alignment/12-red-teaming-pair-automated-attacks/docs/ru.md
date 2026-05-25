# Red-Teaming: PAIR и автоматизированные атаки

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — Prompt Automatic Iterative Refinement — канонический автоматизированный black-box jailbreak. Attacker LLM с red-team system prompt итеративно предлагает jailbreaks для target LLM, накапливая attempts и responses в собственной chat history как in-context feedback. PAIR обычно успешен в пределах 20 queries, на порядки эффективнее GCG (token-level gradient search Zou et al.) и не требует white-box access. PAIR теперь стандартный baseline в JailbreakBench (arXiv:2404.01318) и HarmBench, наряду с GCG, AutoDAN, TAP и Persuasive Adversarial Prompt.

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 minutes

## Цели обучения

- Описать алгоритм PAIR: attacker system prompt, iterative refinement, in-context feedback.
- Объяснить, почему PAIR строго эффективнее GCG, когда target является black-box.
- Назвать четыре других automated-attack baselines (GCG, AutoDAN, TAP, PAP) и указать по одной отличительной особенности каждого.
- Описать evaluation protocols JailbreakBench и HarmBench и то, что означает "attack success rate" в каждом.

## Проблема

Red-teaming раньше был ручной активностью. Небольшое число expert testers конструировало adversarial prompts и отслеживало, какие работают. Это не масштабируется: attack success rate требует статистической выборки, а target является moving target при каждом model release. PAIR операционализирует red-teaming как optimization problem с black-box target.

## Концепция

### Алгоритм PAIR

Inputs:
- Target LLM T (модель, которую мы атакуем).
- Judge LLM J (оценивает, является ли response jailbreak).
- Attacker LLM A (red-team optimizer).
- Goal string G: "respond with [harmful instruction]."
- Budget K (usually 20 queries).

Loop, for k in 1..K:
1. A получает prompt с goal G и history of (prompt, response) pairs so far.
2. A emits a new prompt p_k.
3. Submit p_k to T; receive response r_k.
4. J scores (p_k, r_k) on the goal.
5. If score >= threshold, halt — jailbreak found.
6. Else, append (p_k, r_k) to A's history; continue.

Эмпирический результат (NeurIPS 2023): >50% attack success rate against GPT-3.5-turbo, Llama-2-7B-chat; mean queries to success in the 10-20 range.

### Почему PAIR эффективен

GCG (Zou et al. 2023) ищет adversarial token suffixes через gradient; ему нужен white-box model access, и он производит нечитаемые suffixes. PAIR является black-box и производит natural-language attacks, которые переносятся между моделями. In-context feedback в PAIR позволяет attacker учиться на каждом rejection; у GCG нет эквивалента (каждое новое token update должно заново находить предыдущий прогресс).

### Связанные автоматизированные атаки

- **GCG (Zou et al. 2023, arXiv:2307.15043).** Token-level gradient search for adversarial suffixes. White-box, transferable, produces unreadable strings.
- **AutoDAN (Liu et al. 2023).** Evolutionary search over prompts, guided by a hierarchical objective.
- **TAP (Mehrotra et al. 2024).** Tree-of-attacks with pruning — branches multiple PAIR-style rollouts.
- **PAP (Zeng et al. 2024).** Persuasive Adversarial Prompts — encodes human persuasion techniques as prompt templates.

### JailbreakBench и HarmBench

Оба (2024) стандартизируют evaluation:

- JailbreakBench (arXiv:2404.01318). 100 harmful behaviors across 10 OpenAI-policy categories. Attack success rate (ASR) as the primary metric. Requires a judge (GPT-4-turbo, Llama Guard, or StrongREJECT).
- HarmBench (Mazeika et al. 2024). 510 behaviours across 7 categories, with semantic and functional harm tests. Compares 18 attacks against 33 models.

ASR обычно сообщается при фиксированном query budget. Для сравнения attacks нужно совпадение budgets; 90% ASR at 200 queries не сопоставимы с 85% ASR at 20.

### Почему это важно для deployments 2026 года

Каждая frontier lab теперь запускает PAIR и TAP против production models перед release. ASR trajectories появляются в model cards (Урок 26) и приложениях к safety-case (Урок 18). Атака не экзотична — это стандартная infrastructure.

### Где это находится в Phase 18

Урок 12 — основа automated-attack. Урок 13 (Many-Shot Jailbreaking) — комплементарный length-exploit. Урок 14 (ASCII Art / Visual) — encoding attack. Урок 15 (Indirect Prompt Injection) — production attack surface 2026 года. Урок 16 покрывает defensive-tooling counterparts (Llama Guard, Garak, PyRIT).

## Используйте это

`code/main.py` строит toy PAIR loop. Target — mock classifier, который отказывается от "obvious" harmful prompts (keyword-filter). Attacker — rule-based refiner, который пробует paraphrase, roleplay-framing и encoding. Judge оценивает response. Можно увидеть, как attacker успешен примерно за ~5-15 iterations против keyword filter и терпит неудачу против semantic filter.

## Отгрузите это

Этот урок создает `outputs/skill-attack-audit.md`. По red-team evaluation report он аудирует: какие attacks были запущены (PAIR, GCG, TAP, AutoDAN, PAP), с каким budget каждая, с каким judge, на каком harmful-behaviour set (JailbreakBench, HarmBench, internal).

## Упражнения

1. Запустите `code/main.py`. Измерьте mean-queries-to-success для трех встроенных attacker strategies. Объясните, какое target-defense assumption эксплуатирует каждая.

2. Реализуйте четвертую attacker strategy (например, translation to another language, base64 encoding). Сообщите новый mean-queries-to-success против keyword-filter target и semantic-filter target.

3. Прочитайте Chao et al. 2023 Figure 5 (PAIR vs GCG comparison). Опишите два сценария, где GCG предпочтителен, несмотря на efficiency advantage PAIR.

4. JailbreakBench сообщает ASR против fixed goal set. Спроектируйте дополнительную метрику, измеряющую attack diversity (variance in successful prompts). Объясните, почему diversity важна для defense evaluation.

5. TAP (Mehrotra 2024) расширяет PAIR с branching + pruning. Набросайте TAP-style extension to `code/main.py` и опишите trade-off computational cost vs success-rate.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| PAIR | "automated jailbreak" | Prompt Automatic Iterative Refinement; attacker-LLM + judge-LLM loop |
| GCG | "gradient jailbreak" | White-box token-level gradient search for adversarial suffixes |
| Attack success rate (ASR) | "% jailbreaks at k queries" | Основная метрика; должна сообщаться с query budget и judge identity |
| Judge LLM | "the scorer" | LLM, которая оценивает, удовлетворяет ли response harmful goal |
| JailbreakBench | "the evaluation" | Стандартизированный harmful-behaviour set с tagged categories |
| HarmBench | "the broader bench" | 510 behaviours, functional + semantic harm tests |
| TAP | "tree of attacks" | PAIR с branching + pruning; лучше ASR при большем compute |

## Дополнительное чтение

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — статья PAIR, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — статья GCG
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — стандартизированная evaluation
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — более широкая evaluation
