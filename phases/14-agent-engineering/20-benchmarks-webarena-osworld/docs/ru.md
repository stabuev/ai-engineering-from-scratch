# Бенчмарки: WebArena и OSWorld

> WebArena тестирует возможности веб-агента на четырех самостоятельно развернутых приложениях. OSWorld тестирует возможности desktop-агента на Ubuntu, Windows и macOS. На момент выпуска (2023-2024) оба показывали большой разрыв между лучшими агентами и людьми. Разрыв сокращается; режимы отказа не изменились.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 19 (SWE-bench, GAIA)
**Время:** ~60 минут

## Цели обучения

- Описать четыре самостоятельно развернутых приложения WebArena и объяснить, почему важна оценка на основе выполнения.
- Объяснить, почему OSWorld использует реальные screenshots ОС вместо accessibility APIs.
- Назвать два главных режима отказа OSWorld: GUI grounding и operational knowledge.
- Кратко изложить, что OSWorld-G и OSWorld-Human добавляют поверх базового бенчмарка.

## Проблема

Агенты общего назначения умеют вызывать tools. Могут ли они управлять browser на протяжении 20 clicks, чтобы завершить shopping checkout? Могут ли они настроить Linux box, используя только keyboard и mouse? На эти вопросы отвечают WebArena и OSWorld.

## Концепция

### WebArena (Zhou et al., ICLR 2024)

- 812 long-horizon tasks в четырех самостоятельно развернутых web apps: shopping site, forum, GitLab-like dev tool, business CMS.
- Плюс вспомогательные utilities: map, calculator, scratchpad.
- Evaluation выполняется на основе результата через gym APIs: был ли order placed, issue closed, CMS page updated?
- На момент выпуска лучший GPT-4 agent достиг 14.41% success против 78.24% у человека.

Self-hosted постановка важна: benchmark не flaky, потому что целевые apps зафиксированы и воспроизводимы.

### Расширения

- **VisualWebArena** — visually grounded tasks, где success зависит от интерпретации images (screenshots как наблюдения первого класса).
- **TheAgentCompany** (декабрь 2024) — добавляет terminal + coding; больше похоже на настоящую среду удаленной работы.

### OSWorld (Xie et al., NeurIPS 2024)

- 369 реальных computer tasks на Ubuntu, Windows и macOS.
- Свободное управление keyboard и mouse в реальных applications.
- Screenshots 1920×1080 как observation.
- На момент выпуска лучшая model: 12.24% против 72.36% у человека.

### Основные режимы отказа

1. **GUI grounding.** Pixel → element mapping. Models с трудом надежно локализуют UI elements в 1920×1080.
2. **Operational knowledge.** В каком menu находится setting, какой keyboard shortcut нужен, где лежит preference pane. Длинный хвост знаний, который люди накапливают годами.

### Продолжения

- **OSWorld-G** — grounding suite на 564 samples + Jedi training set. Отделяет grounding от planning, чтобы измерять их отдельно.
- **OSWorld-Human** — вручную подготовленные gold action trajectories. Показывает, что top agents используют в 1.4-2.7x больше steps, чем необходимо (trajectory-efficiency gap).

### Почему это важно

Claude computer use, OpenAI CUA и Gemini 2.5 Computer Use (Урок 21) обучаются на workloads, сформированных WebArena и OSWorld. Benchmarks задают цель; production models дают практический ответ.

### Где benchmarking ломается

- **Screenshot-only evals.** OSWorld screenshot-driven; оценка agent, который использует DOM или accessibility APIs на OSWorld, упускает grounding challenge.
- **Игнорирование trajectory length.** Оценка только success-rate пропускает step inefficiency 1.4-2.7x, которую показывает OSWorld-Human.
- **Устаревшие self-hosted apps.** Apps WebArena закрепляют конкретные versions; update без повторной подготовки ломает сопоставимость.

## Соберите это

`code/main.py` реализует toy web-agent harness:

- Минимальная state machine "shopping app": list_items, add_to_cart, checkout.
- Gold trajectories для 3 tasks.
- Scripted agent, который пытается выполнить каждую task.
- Execution-based evaluator (state check) и trajectory-efficiency metric (steps vs gold).

Запустите:

```
python3 code/main.py
```

Output: success rate по задачам и trajectory efficiency, отражающие методологию OSWorld-Human.

## Используйте это

- **WebArena Verified**, self-hosted на внутреннем cluster для continuous evaluation.
- **OSWorld** в VM fleet для desktop agents.
- **Computer-use agents** (Урок 21) — Claude, OpenAI CUA, Gemini — все обучались на workloads вроде этих.
- **Ваши собственные product flows** — зафиксируйте gold trajectories для top 20 tasks; запускайте agents против них еженедельно.

## Отправьте в работу

`outputs/skill-web-desktop-harness.md` строит web/desktop agent harness с execution-based eval и trajectory efficiency metric.

## Упражнения

1. Расширьте toy harness вторым app (forum). Напишите 3 tasks плюс gold trajectories.
2. Добавьте reporting trajectory-efficiency по task. В вашем toy agent идет в 1x, 2x или 3x сверх gold?
3. Реализуйте "distractor" tool — tool, который gold trajectory никогда не использует. Scripted agent поддается соблазну?
4. Прочитайте OSWorld-G. Как бы вы отделили grounding failures от planning failures в собственных evals?
5. Прочитайте apps README WebArena. Что ломается при upgrade одной из pinned app versions?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| WebArena | "Web agent benchmark" | 812 tasks в 4 self-hosted apps; gym-style evaluation |
| VisualWebArena | "Visual WebArena" | Visually grounded WebArena; screenshots являются observations |
| OSWorld | "Desktop agent benchmark" | 369 tasks на реальных Ubuntu/Windows/macOS |
| GUI grounding | "Pixel-to-element mapping" | Model локализует UI elements в 1920x1080 |
| Operational knowledge | "OS know-how" | Какое menu, какой shortcut, какая preference pane нужны |
| OSWorld-G | "Grounding suite" | 564 grounding-only samples + training set |
| OSWorld-Human | "Gold trajectories" | Ручные expert action sequences для измерения efficiency |
| Trajectory efficiency | "Steps over gold" | Agent step count, деленный на human minimum |

## Дополнительное чтение

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — web benchmark из четырех apps
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — cross-OS desktop benchmark
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — benchmark-shaped capability Claude
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — числа OSWorld и WebArena
