# Библиотеки навыков и lifelong learning (Voyager)

> Voyager (Wang et al., TMLR 2024) рассматривает исполняемый код как skill. Skills именованы, доступны через retrieval, компонуемы и уточняются по feedback от среды. Это эталонная архитектура для Claude Agent SDK skills, skillkit и паттерна skill-library 2026 года.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 07 (MemGPT), Фаза 14 · 08 (Letta Blocks)
**Время:** ~75 минут

## Цели обучения

- Назвать три компонента Voyager — automatic curriculum, skill library, iterative prompting — и роль каждого.
- Объяснить, почему Voyager делает пространством действий код, а не примитивные команды.
- Реализовать stdlib-библиотеку skills с регистрацией, retrieval, композицией и refinement по сбоям.
- Сопоставить паттерн Voyager со skills Claude Agent SDK 2026 года и экосистемой skillkit.

## Проблема

Агенты, которые заново строят каждую способность в каждой сессии, делают три вещи неправильно:

1. **Тратят токены.** Каждая задача заново eliciting одно и то же reasoning.
2. **Теряют прогресс.** Исправление, выученное в session A, не переносится в session B.
3. **Проваливаются на long-horizon composition.** Сложным задачам нужны иерархии возможностей; one-shot prompts не могут их выразить.

Ответ Voyager: рассматривать каждую переиспользуемую способность как именованный фрагмент кода, сохраненный в библиотеке, доступный по similarity, компонуемый с другими skills и уточняемый по execution feedback.

## Концепция

### Три компонента

Voyager (arXiv:2305.16291) строит агента вокруг:

1. **Automatic curriculum.** Curiosity-driven proposer выбирает следующую задачу на основе текущего набора skills агента и состояния среды. Exploration идет снизу вверх.
2. **Skill library.** Каждый skill — исполняемый код. Новые skills добавляются, когда задача успешна. Skills извлекаются по similarity между query и description.
3. **Iterative prompting mechanism.** При сбое агент получает execution errors, environment feedback и self-verification output, затем уточняет skill.

Оценка в Minecraft (Wang et al., 2024): в 3.3x больше уникальных предметов, в 8.5x быстрее каменные инструменты, в 6.4x быстрее железные инструменты, в 2.3x длиннее обход карты по сравнению с baselines. Числа специфичны для Minecraft, но паттерн переносится.

### Action space = code

Большинство агентов выдают примитивные команды. Voyager выдает JavaScript functions. Skill выглядит так:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Скомпонован из sub-skills. Хранится с ключом по description и embedding. Извлекается как программа, а не как prompt.

Это skill Claude Agent SDK 2026 года: именованный, доступный через retrieval фрагмент кода плюс инструкции, которые агент загружает по требованию.

### Skill retrieval

Новая задача "make a diamond pickaxe." Агент:

1. Embeds описание задачи.
2. Запрашивает в skill library top-k похожих skills.
3. Извлекает `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` etc.
4. Компонует новый skill из retrieved primitives + новой логики.

Это паттерн, который реализуют MCP resources (Phase 13) и Agent SDK skills: retrieval по поверхности knowledge/code, scoped к текущей задаче.

### Iterative refinement

Feedback loop Voyager:

1. Агент пишет skill.
2. Skill запускается в среде.
3. Возвращается один из трех сигналов: `success`, `error` (со stack trace), `self-verification failure`.
4. Агент переписывает skill, используя сигнал как context.
5. Цикл идет до success или max rounds.

Это Self-Refine (Lesson 05), примененный к генерации кода с environment-grounded verification. CRITIC (Lesson 05) — тот же паттерн с внешними tools как verifier.

### Curriculum и exploration

Модуль curriculum в Voyager предлагает задачи вроде "build a shelter near the lake" на основе того, что у агента есть и чего он еще не делал. Proposer использует состояние среды + inventory skills, чтобы выбрать задачу чуть выше текущей capability — sweet spot exploration.

Для продакшен-агентов это превращается в оператор "what's missing": given текущая skill library and domain, какие skills еще не покрыты? Команды обычно реализуют это вручную как curriculum review.

### Где этот паттерн ломается

- **Skill library rot.** Один и тот же skill добавлен 10 раз с немного разными descriptions. Добавьте deduplication on write; retrieval возвращает только один.
- **Composed-skill drift.** Parent skill зависит от child, который был refined. Версионируйте skills; parent, pinned to v1, не должен магически подхватить v3.
- **Retrieval quality.** Vector retrieval по skill descriptions деградирует, когда библиотека перерастает несколько сотен. Дополните его tag filters и hard constraints ("only skills with `category=tooling`").

## Соберите это

`code/main.py` реализует stdlib skill library:

- `Skill` — name, description, code (as string), version, tags, dependencies.
- `SkillLibrary` — register, search (token overlap), compose (topological sort of deps) и refine (version bump on update).
- Скриптовый агент, который регистрирует три primitive skills, компонует четвертый, ловит failure и делает refine.

Запустите:

```
python3 code/main.py
```

Трасса показывает записи в библиотеку, retrieval, composition, failed execution и v2 refinement — цикл Voyager end to end.

## Используйте это

- **Claude Agent SDK skills** (Anthropic) — эталон 2026 года: у каждого skill есть description, code и instructions; загружается по требованию во время agent session.
- **skillkit** (npm: skillkit) — управление skills между agents для 32+ AI coding agents.
- **Custom skill libraries** — domain-specific (SQL skills для data agents, Terraform skills для infra agents). Паттерн Voyager масштабируется вниз.
- **OpenAI Agents SDK `tools`** — на нижнем уровне; каждый tool является lightweight skill.

## Доведите до продакшена

`outputs/skill-skill-library.md` генерирует Voyager-образную skill library с registration, retrieval, versioning и refinement для любого target runtime.

## Упражнения

1. Добавьте detector циклов зависимостей в `compose()`. Что происходит, когда skill A зависит от B, который зависит от A? Error vs warning?
2. Реализуйте per-skill version pinning. Когда parent skill компонует child `crafting@1`, refinement до `crafting@2` не должен silently upgrade parent.
3. Замените token-overlap retrieval на sentence-transformers embeddings (или BM25 stdlib impl). Измерьте retrieval@5 на игрушечной библиотеке из 50 skills.
4. Добавьте "curriculum" agent: given текущая library and domain description, предложить 5 missing skills. Вызывайте еженедельно.
5. Прочитайте документацию Anthropic по Claude Agent SDK skills. Перенесите игрушечную library в skill schema SDK. Что меняется в discoverability?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Skill | "Переиспользуемая способность" | Именованный фрагмент кода + description, извлекаемый по similarity |
| Skill library | "Agent memory of how-to" | Постоянное хранилище skills, searchable и composable |
| Curriculum | "Task proposer" | Bottom-up генератор целей, driven текущим capability gap |
| Composition | "Skill DAG" | Skills вызывают skills; topologically sorted on execution |
| Iterative refinement | "Self-correcting loop" | Env feedback + errors + self-verification возвращаются в следующую версию |
| Action-space-as-code | "Programmatic actions" | Выдавать functions, а не primitive commands, для temporally extended behavior |
| Dedup on write | "Skill collapse" | Почти дублирующиеся descriptions схлопываются в один canonical skill |

## Дополнительное чтение

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — исходная статья о skill-library
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — skills как productization 2026 года
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skills и subagents на практике
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — refinement loop под Voyager
