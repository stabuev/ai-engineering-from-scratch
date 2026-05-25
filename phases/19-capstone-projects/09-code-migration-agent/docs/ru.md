# Capstone 09 — Агент миграции кода (обновление языка / runtime на уровне репозитория)

> Amazon MigrationBench (Java 8 to 17) и мигратор Google App Engine Py2-to-Py3 задают планку 2026 года. Moderne OpenRewrite выполняет детерминированные AST-переписывания в масштабе. Grit решает ту же задачу через DSL в стиле codemod. Продакшен-паттерн объединяет оба подхода: детерминированная основа для безопасных переписываний, агентный слой для неоднозначных случаев, sandbox для сборок по веткам и test harness, который становится зеленым до открытия PR. Капстоун: мигрировать 50 реальных репозиториев и опубликовать pass rate с таксономией отказов.

**Тип:** Capstone
**Языки:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Предварительные требования:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Отрабатываемые фазы:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Время:** 30 часов

## Проблема

Крупномасштабная миграция кода - одно из самых чистых продакшен-применений coding agents в 2026 году. Ground truth очевиден (проходит ли test suite после миграции?), выгода реальна (миграция парка Java 8 - проект масштаба команды), а бенчмарки публичны (подмножество MigrationBench из 50 repos). Moderne OpenRewrite закрывает детерминированную часть. Агентный слой обрабатывает все, с чем не справляются OpenRewrite recipes: неоднозначные переписывания, дрейф build-system, long-tail синтаксис, поломки транзитивных зависимостей.

Вы построите агента, который принимает Java 8 repo (или Python 2 repo) и создает мигрированную ветку с зеленым CI. Вы измерите pass rate, сохранение test coverage, cost per repo и построите таксономию отказов. Сравнение бок о бок с baseline "deterministic-only" покажет, где на самом деле находится ценность агента.

## Концепция

Pipeline состоит из двух слоев. **Детерминированная основа** (OpenRewrite для Java, libcst для Python) безопасно выполняет основную массу механических переписываний: imports, method signatures, null-safety edits, try-with-resources, deprecated API replacements. Она быстрая и создает аудируемые diffs. **Агентный слой** (OpenAI Agents SDK или LangGraph поверх Claude Opus 4.7 и GPT-5.4-Codex) обрабатывает случаи, которые recipes не покрывают: обновления build files (Maven/Gradle/pyproject), конфликты транзитивных зависимостей, test flakes, custom annotations.

Каждый repo получает Daytona sandbox с предустановленным целевым runtime. Агент итерирует: запускает build, классифицирует failures, применяет fix, запускает повторно. Жесткие лимиты: 30 минут на repo, $8 на repo, 20 agent turns. Если все тесты проходят и coverage delta не отрицательная, ветка открывает PR. Если нет, repo попадает в класс отказа с evidence.

Таксономия отказов - главный результат. Что сломалось на 50 repos? Transitive deps? Custom annotations? Build tool version? Test flakes, не связанные с миграцией? Каждый класс получает количество случаев и exemplar diff. Будущие авторы recipes смогут нацелиться на три главные категории.

## Архитектура

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Стек

- Детерминированная основа: OpenRewrite (Java) или libcst (Python)
- Агент: OpenAI Agents SDK или LangGraph поверх Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainers по ветке, предустановленный целевой runtime (Java 17 / Python 3.12)
- Системы сборки: Maven, Gradle, uv (Python)
- Бенчмарки: Amazon MigrationBench 50-repo subset (Java 8 to 17), Google App Engine Py2-to-Py3 repos
- Test harness: parallel runner, coverage через Jacoco (Java) или coverage.py (Python)
- Observability: Langfuse + trace bundle per repo с каждым diff chunk
- Dashboard: dashboard таксономии отказов с количеством случаев по классам и exemplar diffs

## Соберите

1. **Проход recipes.** Сначала запустите OpenRewrite (Java) или libcst (Python) recipes. Захватите 70-80% миграций, которые являются механическими. Закоммитьте как "recipe" commit.

2. **Пробная сборка.** Daytona sandbox: установите target runtime, запустите build. Если build зеленый, переходите к tests. Если красный, передайте агенту.

3. **Agent loop.** LangGraph с tools: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. Агент классифицирует failure (dep, syntax, test, build-tool) и применяет точечный fix. Запустите повторно.

4. **Бюджетные ограничения.** 30 минут wall-clock на repo, $8 cost, 20 agent turns. Любое превышение останавливает процесс и помещает repo в "budget_exhausted" с текущим diff.

5. **Gate тестов и coverage.** После того как build стал зеленым, запустите test suite. Сравните coverage с base repo. Если coverage упала больше чем на 2%, поместите в "coverage_regression".

6. **Открытие PR.** При успехе отправьте branch, откройте PR с diff и кратким описанием того, какие recipes применились и какие commits написал агент.

7. **Таксономия отказов.** Для каждого failed repo добавьте tag с классом: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Постройте dashboard.

8. **Прогон на 50 repos.** Выполните на подмножестве MigrationBench. Отчитайтесь по per-class pass rate, cost-per-repo, coverage-preservation и сравнению с deterministic-only baseline.

## Используйте

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Сдайте

`outputs/skill-migration-agent.md` - deliverable. Для заданного repo он выполняет deterministic recipes, затем agent loop, чтобы создать зеленую мигрированную ветку, либо помещает repo в класс таксономии.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | MigrationBench pass rate | pass@1 на подмножестве из 50 repos |
| 20 | Сохранение test coverage | Средняя coverage delta относительно base |
| 20 | Cost per migrated repo | $/repo на successful runs |
| 20 | Интеграция агента и deterministic-tool | Доля fixes, которые обработал OpenRewrite, относительно fixes, написанных агентом |
| 15 | Анализ отказов | Полнота таксономии с exemplars |
| **100** | | |

## Упражнения

1. Запустите migrate pipeline только с OpenRewrite (без агента). Сравните pass rate с полным pipeline. Определите случаи, где именно agent меняет исход.

2. Реализуйте проверку "lint-clean": после миграции запустите style linter (spotless для Java, ruff для Python). Проваливайте PR, если появляются новые lint errors. Измерьте coverage-preserved-but-style-regressed rate.

3. Добавьте optimizer "minimal-diff": после того как branch агента проходит tests, обрежьте ненужные изменения вторым проходом. Отчитайтесь о сокращении diff-size.

4. Расширьте на третью миграцию: Node 18 to Node 22. Переиспользуйте sandbox wrapping; замените recipe layer на custom codemod.

5. Измерьте time-to-first-green-build (TTFGB) как UX metric. Цель: p50 меньше 10 минут.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Deterministic substrate | "Recipe engine" | OpenRewrite / libcst: декларативные AST-переписывания с гарантиями безопасности |
| Codemod | "Code-modifying program" | Rewrite rule, который механически меняет source code |
| Build drift | "Tool version skew" | Тонкие изменения поведения Maven / Gradle / uv между major versions |
| Failure class | "Taxonomy bucket" | Маркированная причина, по которой repo не мигрировал: dep, syntax, test, build-tool, budget |
| Coverage delta | "Coverage preservation" | Изменение test coverage % от base к migrated branch |
| Agent turn | "Tool-call round" | Один цикл plan -> act -> observe в agent loop |
| Budget exhaustion | "Hit the ceiling" | Repo израсходовал лимит 30-min / $8 / 20-turn, так и не пройдя tests |

## Дополнительное чтение

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — канонический benchmark 2026 года
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — reference для детерминированной основы
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — альтернативный codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — reference для Agents SDK
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — альтернативный migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — deterministic substrate для Python
- [Daytona sandboxes](https://daytona.io) — reference для sandbox по ветке
