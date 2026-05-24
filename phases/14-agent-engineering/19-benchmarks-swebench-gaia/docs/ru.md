# Бенчмарки: SWE-bench, GAIA, AgentBench

> Три benchmarks задают опору для оценки agents в 2026 году. SWE-bench тестирует code patching. GAIA тестирует generalist tool use. AgentBench тестирует reasoning в нескольких environments. Знайте их состав, историю contamination и то, что они не измеряют.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 06 (Tool Use)
**Время:** ~60 минут

## Цели обучения

- Назвать test harness SWE-bench (FAIL_TO_PASS) и объяснить, почему он gate на unit tests.
- Объяснить, почему существует SWE-bench Verified (OpenAI, 500 tasks) и что он убирает.
- Описать дизайн GAIA: просто для людей, трудно для AI; три уровня сложности.
- Назвать восемь environments AgentBench и главный blocker для open-source LLMs.
- Кратко изложить вывод SWE-bench+ о contamination и его implications.

## Проблема

Leaderboards говорят, какая model выигрывает на одном benchmark. Они не говорят:

- Contaminated ли benchmark (solutions в training data, test leakage).
- Измеряет ли benchmark то, что вам важно (code vs browsing vs generalist).
- Надежен ли evaluator (AST matching, state checks, human review).

Знайте три anchoring benchmarks и их failure modes, прежде чем цитировать число.

## Концепция

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- 2,294 реальных GitHub issues из 12 популярных Python repos.
- Agent получает: codebase на pre-fix commit + natural-language issue description.
- Agent производит: patch.
- Evaluator: применить patch, запустить test suite repo. Patch должен перевести FAIL_TO_PASS tests (раньше падали, теперь проходят) без поломки PASS_TO_PASS tests.

SWE-agent (Yang et al., 2024) достиг 12.5% на release, сделав акцент на agent-computer interfaces (file editor commands, search syntax, понятный модели).

### SWE-bench Verified

OpenAI, август 2024. Human-curated subset из 500 tasks. Убирает ambiguous issues, unreliable tests и tasks, где fix был неясен. Основной benchmark для вопроса "does your agent ship real patches?"

### Contamination

- Более 94% issues SWE-bench появились до cutoffs большинства models.
- **SWE-bench+** обнаружил, что 32.67% successful patches имели leaked solutions в issue text (model видела fix в description), а 31.08% были suspicious из-за weak test coverage.
- Verified чище, но не свободен от contamination.

Практическое следствие: model, которая набирает 50% на SWE-bench, может набрать 35% на SWE-bench+. Всегда сообщайте оба числа, если заявляете performance на SWE-bench.

### GAIA (Mialon et al., ноябрь 2023)

- 466 questions; 300 retained для private leaderboard на huggingface.co/gaia-benchmark.
- Философия дизайна: "conceptually simple for humans (92%) but hard for AI (GPT-4 with plugins: 15%)."
- Тестирует reasoning, multi-modality, web, tool use.
- Три уровня сложности; Level 3 требует длинных tool chains across modalities.

GAIA запускают, чтобы измерить "generalist capability". Не путайте с code-specific benchmarks.

### AgentBench (Liu et al., ICLR 2024)

- 8 environments по code (Bash, DB, KG), games (Alfworld, LTP), web (WebShop, Mind2Web) и open-ended generation.
- Multi-turn, ~4k-13k turns на split.
- Главный вывод: long-term reasoning, decision-making и instruction following — blockers для OSS LLMs, которые догоняют commercial.

### Что они не измеряют

- Реальную operational cost (tokens, wall-clock).
- Safety behavior в adversarial conditions.
- Performance на вашем domain (используйте собственные evals, Урок 30).
- Tail failures (benchmarks усредняют; production operators волнует худший 1%).

### Где benchmarking ломается

- **Single-number fixation.** SWE-bench 50% говорит меньше, чем P50/P75/P95 cost + step distribution.
- **Contaminated claims.** Отчет о SWE-bench без упоминания Verified или SWE-bench+ вводит в заблуждение.
- **Benchmark-as-development-target.** Оптимизация под benchmark расходится с production usefulness.

## Соберите это

`code/main.py` реализует toy harness в стиле SWE-bench:

- Synthetic bug-fix tasks (3 tasks).
- Scripted "agent", который предлагает patches.
- Test runner, который проверяет FAIL_TO_PASS (bug теперь исправлен) и PASS_TO_PASS (ничего не сломано).
- GAIA-style difficulty classifier на основе глубины decomposition question.

Запустите:

```
python3 code/main.py
```

Output показывает resolution rate по task + по difficulty и делает правила evaluator конкретными.

## Используйте это

- **SWE-bench Verified** для code agents. Всегда сообщайте Verified scores.
- **GAIA** для generalist agents. Используйте private leaderboard split.
- **AgentBench** для multi-environment comparison.
- **Custom evals** (Урок 30) для фактической формы вашего продукта.

## Отправьте в работу

`outputs/skill-benchmark-harness.md` строит SWE-bench-style harness для любой пары codebase-task с gating FAIL_TO_PASS / PASS_TO_PASS.

## Упражнения

1. Перенесите toy harness на реальный repo (выберите один из своих). Напишите 3 FAIL_TO_PASS tests для известных bugs.
2. Добавьте step-count metric. На ваших 3 tasks сколько agent steps приходится на resolution?
3. Прочитайте paper SWE-bench+. Реализуйте solution-leakage check (pattern-match issue text against diff).
4. Скачайте GAIA question из public split. Проследите, что сделал бы GPT-4-class agent. Какие tools ему нужны?
5. Прочитайте per-environment breakdown AgentBench. Какая environment зеркалит поверхность вашего продукта? Как там выглядит "SOTA"?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| SWE-bench | "Code agent benchmark" | 2,294 GitHub issues; patch должен перевести FAIL_TO_PASS tests |
| SWE-bench Verified | "Чистый SWE-bench" | 500 human-curated tasks, OpenAI |
| FAIL_TO_PASS | "Fix gate" | Tests, которые раньше падали и должны проходить после patch |
| PASS_TO_PASS | "No-regression gate" | Tests, которые проходили и должны продолжать проходить |
| GAIA | "Generalist benchmark" | 466 human-easy / AI-hard multi-tool questions |
| AgentBench | "Multi-env benchmark" | 8 environments; long-horizon multi-turn |
| Contamination | "Training-set leak" | Benchmark tasks, присутствующие в model training |
| SWE-bench+ | "Contamination audit" | 32.67% solution leakage найдено в successful SWE-bench patches |

## Дополнительное чтение

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — исходный benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — generalist benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — multi-environment suite
