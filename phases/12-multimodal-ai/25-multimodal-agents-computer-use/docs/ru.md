# Мультимодальные агенты и computer-use (Capstone)

> Frontier-продукт 2026 года — мультимодальный агент, который читает screenshots, кликает buttons, навигирует web UIs, заполняет forms и выполняет workflows end-to-end. SeeClick и CogAgent (2024) доказали примитив GUI-grounding. Ferret-UI добавил mobile. ChartAgent ввел visual tool-use для charts. VisualWebArena и AgentVista (2026) — бенчмарки, за которыми гонится frontier, и даже Gemini 3 Pro и Claude Opus 4.7 набирают ~30% на hard tasks AgentVista. Этот capstone собирает все линии Phase 12: perception (high-res VLM), reasoning (LLM with tool use), grounding (coordinate output), long-horizon memory и evaluation.

**Тип:** Capstone
**Языки:** Python (stdlib, action schema + agent loop skeleton)
**Предварительные требования:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Время:** ~240 минут

## Цели обучения

- Спроектировать multimodal agent loop: perceive → reason → act → observe → repeat.
- Построить GUI grounding output schema (click coordinates, type text, scroll, drag), которую VLM может выдавать как JSON.
- Сравнить screenshot-only agents, accessibility-tree agents и hybrid agents.
- Настроить multimodal agent benchmark evaluation на небольшом срезе VisualWebArena.

## Проблема

Workflow сайта бронирования: "find me a flight to Tokyo for April 15, aisle seat under $800, book it."

Мультимодальному агенту нужно:

1. Сделать screenshot браузера.
2. Разобрать screenshot + URL + goal в plan.
3. Выдать structured action: click (at x,y), type "Tokyo" (at element E), scroll down, select (radio button).
4. Применить action к браузеру.
5. Наблюдать новое state (следующий screenshot).
6. Повторять, пока задача не выполнена.

Каждый шаг — мультимодальный VLM call. Выход VLM должен быть parseable JSON. Ошибки накапливаются по шагам, поэтому recovery важен.

## Концепция

### GUI grounding — примитив

GUI grounding: по screenshot и инструкции на естественном языке выдать координату (x, y), куда кликнуть (или другое action).

SeeClick (arXiv:2401.10935) был первым открытым результатом в масштабе: fine-tune VLM на synthetic + real GUI data, output coordinates как plain text tokens. Работает.

CogAgent (arXiv:2312.08914) добавил high-resolution encoding 1120x1120 для dense UIs. Score: ~84% on web navigation.

Ferret-UI (arXiv:2404.05719) фокусируется на mobile UIs, интегрируется с iOS accessibility data.

Output format обычно JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` помогает recovery: если coordinates drift между screenshots, semantic hint позволяет системе re-ground.

### Action schemas

Типичная action schema имеет 6-10 action types:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Агент выдает одно action на step. Browser wrapper выполняет его и возвращает новое state.

### Screenshot-only vs accessibility-tree

Два входных режима:

- Screenshot-only: полное image, без структурной информации. Самый общий; работает в любом app.
- Accessibility tree: структурированная DOM / iOS accessibility info. Намного надежнее для grounding; работает там, где tree доступно.
- Hybrid: оба, с tree как надежным grounder для atomic actions и screenshot для semantic context.

Production agents используют hybrid, когда возможно. Browser automation (Selenium + accessibility) всегда имеет tree; desktop apps иногда имеют.

### Long-horizon memory

Workflow на 20 шагов создает 20 screenshots. Context VLM быстро заполняется. Три стратегии сжатия:

- Summary-chain: после каждых 5 steps суммировать, что произошло, удалить old screenshots.
- Skip-frame: хранить first, last и every 3rd screenshot.
- Tool-recorded log: выполнять actions, хранить text log сделанного; не пересматривать old screenshots.

Claude's computer-use API использует log pattern. Проще и надежнее.

### Visual tool use

ChartAgent (arXiv:2510.04514) вводит visual tool use для chart understanding: crop, zoom, OCR, call external detection. Агент может выдать "crop to region (100, 200, 300, 400) then call OCR" как tool call. Tool возвращает text; VLM продолжает reasoning.

Этот паттерн обобщается: set-of-mark prompting, region annotation и external detection tools укладываются в одну схему "output a tool call, receive a structured response".

### Бенчмарки 2026

- ScreenSpot-Pro. GUI grounding на ~1k web screenshots. Open SOTA Qwen2.5-VL-72B ~85%. Frontier ~90%.
- VisualWebArena. End-to-end web tasks (shop, forum, classifieds). Open SOTA ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). Самый сложный benchmark 2026. Realistic workflows across 12 domains. Frontier models score 27-40%; open models 10-20%.
- WebArena / WebShop. Старые benchmarks; saturated by frontier.

### Почему это все еще сложно

Узкие места производительности агента:

1. Visual grounding at fine scale. "Click the small X" часто проваливается на mobile resolution.
2. Long-horizon planning. После 10 actions агент отклоняется от goal.
3. Error recovery. Когда click fails (wrong button), detecting + recovering редко присутствует в trained data.
4. Cross-page context. Переходы между tabs или long forms теряют state.

Направления исследований: memory architectures, explicit replanning, multimodal verification (screenshot match for action success).

### Capstone build-it

Задача capstone: построить computer-use agent, который:

1. Читает HTML + screenshot mock page сайта бронирования.
2. Планирует multi-step sequence: search → select → fill form → submit.
3. Выдает JSON actions, matching the action schema.
4. Оценивается на fixed 10-task slice.

Урок предоставляет scaffold code, который легко расширить до реального browser.

## Применение

`code/main.py` — capstone scaffold:

- Action schema JSON definition (10 actions).
- Mock browser state as dict.
- Agent loop skeleton: receive state, emit action, apply, loop.
- 10-task mini-benchmark (synthetic pages) для измерения end-to-end success rate.
- Error-recovery hook на случай, когда action fails.

## Результат

Этот урок создает `outputs/skill-multimodal-agent-designer.md`. По computer-use product (domain, action set, evaluation target) проектирует полный agent loop, memory strategy, grounding mode и expected benchmark score.

## Упражнения

1. Расширьте action schema инструментом `screenshot_region` (crop + zoom). Какие tasks получают пользу?

2. Прочитайте AgentVista (arXiv:2602.23166). Опишите самую сложную task category и почему frontier models все еще fail.

3. Long-horizon memory compression: спроектируйте summary-chain с ≤4 screenshots kept live, any number logged.

4. Постройте error-recovery hook: при action failure (button not found), что агент делает дальше?

5. Сравните screenshot-only Claude 4.7 с hybrid screenshot + accessibility-tree Qwen2.5-VL на 10 web tasks. Какая система выигрывает на каких tasks?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| GUI grounding | "Click coordinates" | Модель выдает (x,y) для цели инструкции на screenshot |
| Action schema | "Tool definitions" | JSON description of valid actions (click, type, scroll, drag) |
| Accessibility tree | "Structured DOM" | Машиночитаемая UI hierarchy из browser/iOS APIs |
| Hybrid agent | "Screenshot + tree" | Использует и image, и structured info; надежнее каждого по отдельности |
| Visual tool use | "Zoom/crop/detect" | Агент вызывает external vision tools (OCR, detection) в середине плана |
| Summary-chain | "Memory compression" | Periodic text summaries заменяют long screenshot history |
| VisualWebArena | "E2E web bench" | Benchmark 2024 для end-to-end web tasks |
| AgentVista | "2026 hard bench" | 12-domain realistic workflows; даже Gemini 3 Pro набирает ~30% |

## Дополнительное чтение

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
