# Red-Team Tooling — Garak, Llama Guard, PyRIT

> Три production tools задают red-team stack 2026 года. Llama Guard (Meta) — Llama-3.1-8B classifier, дообученный на 14 MLCommons hazard categories; Llama Guard 4 2025 года — 12B natively multimodal classifier, pruned from Llama 4 Scout. Garak (NVIDIA) — open-source LLM vulnerability scanner со static, dynamic и adaptive probes для hallucination, data leakage, prompt injection, toxicity и jailbreaks. PyRIT (Microsoft) — multi-turn red-team campaigns с Crescendo, TAP и custom converter chains для deep exploitation. Llama Guard 3 задокументирован в Meta's "Llama 3 Herd of Models" (arXiv:2407.21783); Llama Guard 3-1B-INT4 в arXiv:2411.17713; probe architecture Garak в github.com/NVIDIA/garak. Эти инструменты — production interface 2026 года между red-team research (Lessons 12-15) и deployment (Lesson 17+).

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## Цели обучения

- Описать место Llama Guard 3/4 в safety stack: input classifier, output classifier или оба.
- Назвать 14 MLCommons hazard categories и одну неочевидную категорию (Code Interpreter Abuse).
- Описать probe architecture Garak: probes, detectors, harnesses.
- Описать multi-turn campaign structure PyRIT и то, как она сочетается с Garak probes.

## Проблема

Lessons 12-15 представляют поверхность атаки. Production deployments нуждаются в повторяемой, масштабируемой evaluation. В 2026 году доминируют три инструмента: Llama Guard (defense classifier), Garak (scanner), PyRIT (campaign orchestrator). Каждый нацелен на отдельный слой red-team lifecycle.

## Концепция

### Llama Guard (Meta)

Llama Guard 3 — модель Llama-3.1-8B, дообученная для input/output classification по 14 категориям MLCommons AILuminate:
- Violent crimes, non-violent crimes, sex-related, CSAM, defamation
- Specialized advice, privacy, IP, indiscriminate weapons, hate
- Suicide/self-harm, sexual content, elections, code-interpreter abuse

Поддерживает 8 languages. Использование: поставить перед LLM (input moderation), после LLM (output moderation) или в обоих местах. Эти два применения создают разные training distributions — Llama Guard 3 поставляется как единая модель, обрабатывающая оба.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 tokens/s on mobile CPU) — квантованный edge variant.

Llama Guard 4 (April 2025) — 12B, natively multimodal, pruned from Llama 4 Scout. Она заменяет и 8B text, и 11B vision predecessors одним классификатором, который принимает text + images.

### Garak (NVIDIA)

Open-source vulnerability scanner. Architecture:
- **Probes.** Генераторы атак для hallucination, data leakage, prompt injection, toxicity, jailbreaks. Static (fixed prompts), dynamic (generated prompts), adaptive (responds to target output).
- **Detectors.** Оценивают outputs относительно expected failure modes — toxic, leaked, jailbroken.
- **Harnesses.** Управляют probe-detector pairs, запускают campaigns, генерируют reports.

TrustyAI интегрирует Garak с Llama-Stack shields (Prompt-Guard-86M input classifier, Llama-Guard-3-8B output classifier) для end-to-end shielded-target evaluation. Tier-based scoring (TBSA) заменяет binary pass/fail — model может pass на severity tier 3 и fail на severity tier 5 на том же probe.

### PyRIT (Microsoft)

Python Risk Identification Toolkit. Multi-turn red-team campaigns. Построен вокруг:
- **Converters.** Преобразуют seed prompt — paraphrase, encode, translate, roleplay.
- **Orchestrators.** Запускают campaign: Crescendo (escalation), TAP (branching), RedTeaming (custom loop).
- **Scoring.** LLM-as-judge или classifier-as-judge.

PyRIT — более тяжелый родственник Garak. Garak запускает тысячи single-turn probes; PyRIT запускает глубокие multi-turn campaigns, спроектированные для взлома конкретных failure modes.

### The stack

Поставьте Llama Guard по обе стороны модели. Запускайте Garak nightly для regression. Запускайте PyRIT для pre-release campaigns. Это default configuration 2026 года для большинства production deployments.

### Evaluation pitfalls

- **Judge identity.** Все три инструмента могут использовать LLM judge; judge calibration определяет reported ASRs (Lesson 12). Указывайте judge вместе с tool.
- **Probe staleness.** Garak probes устаревают по мере того, как модели patch against them. Adaptive probes (PAIR-shaped) устаревают медленнее, чем static probes.
- **Llama Guard FPR on benign content.** Ранние версии Llama Guard чрезмерно помечали political и LGBTQ+ content; calibrations Llama Guard 3/4 улучшены, но не calibrated per-deployment.

### Как это вписывается в Phase 18

Lessons 12-15 — семейства атак. Lesson 16 — production tooling. Lesson 17 (WMDP) — evaluation для dual-use capability. Lesson 18 — frontier safety frameworks, которые оборачивают эти инструменты в policy structure.

## Применение

`code/main.py` строит игрушечный Llama Guard-style classifier (keyword + semantic features over 14 categories), игрушечный Garak harness (probe-detector loop) и PyRIT-style multi-turn converter chain. Вы можете запустить три инструмента против mock target и наблюдать разные coverage signatures.

## Результат

Этот урок создает `outputs/skill-red-team-stack.md`. Для описания deployment он называет, какие из трех инструментов уместны, что настроить в каждом и какую regression cadence запускать.

## Упражнения

1. Запустите `code/main.py`. Сравните detection rate Llama-Guard-style classifier на single-turn vs multi-turn attacks.

2. Реализуйте новый Garak probe: base64-encoded harmful request. Измерьте его обнаружение Llama-Guard-style classifier.

3. Расширьте PyRIT-style converter chain с помощью converter "translate to French, then paraphrase". Повторно измерьте attack success.

4. Прочитайте hazard-category list Llama Guard 3. Определите две категории, где training data реалистично даст высокие false-positive rates на легитимном developer content.

5. Сравните design principles Garak и PyRIT. Обоснуйте deployment, где каждый является правильным инструментом.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Llama Guard | "the classifier" | Fine-tuned Llama-3.1-8B/4-12B safety classifier с 14 hazard categories |
| Garak | "the scanner" | NVIDIA open-source vulnerability scanner; probes, detectors, harnesses |
| PyRIT | "the campaign tool" | Microsoft multi-turn red-team orchestrator; converters, orchestrators, scoring |
| Prompt-Guard | "the small classifier" | Meta's 86M prompt-injection classifier, paired with Llama Guard |
| TBSA | "tier-based scoring" | Garak's tier-based pass/fail, заменяющий binary outcomes |
| Converter chain | "paraphrase + encode + ..." | PyRIT composition primitive для построения multi-step attacks |
| MLCommons hazard categories | "the 14 taxonomies" | Industry-standard taxonomy, на которую нацелен Llama Guard |

## Дополнительное чтение

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — классификатор 8B
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — квантованный mobile classifier
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — scanner repo and documentation
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — campaign toolkit
