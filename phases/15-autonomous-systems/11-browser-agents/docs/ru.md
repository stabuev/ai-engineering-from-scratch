# Браузерные агенты и long-horizon web tasks

> ChatGPT agent (июль 2025) объединил Operator и deep research в одного browser/terminal agent и установил BrowseComp SOTA на 68.9%. OpenAI закрыла Operator 31 августа 2025 — консолидация на продуктовом слое. Приобретение Vercept компанией Anthropic подняло Claude Sonnet на OSWorld с менее чем 15% до 72.5%. WebArena-Verified (ServiceNow, ICLR 2026) исправил 11.3 процентных пункта false-negative rate в исходном WebArena и выпустил Hard subset из 258 задач. Числа реальны. Поверхность атаки тоже: руководитель OpenAI preparedness публично заявил, что indirect prompt injection в browser agents «не является багом, который можно полностью исправить». Задокументированные атаки 2025–2026: Tainted Memories (Atlas CSRF), HashJack (Cato Networks) и one-click hijacks в Perplexity Comet.

**Тип:** Обучение
**Языки:** Python (stdlib, модель поверхности атаки indirect prompt injection)
**Предварительные требования:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Время:** ~45 минут

## Цели обучения

- Прослеживать линию браузерных агентов (Operator → ChatGPT agent) и SOTA на BrowseComp.
- Объяснять риски доверия и безопасности браузерного / терминального агента.
- Определять, что браузерный агент делает надёжно, а что нет.

## Проблема

Browser agent — это long-horizon agent, который читает недоверенный контент и выполняет значимые действия. Каждая страница, которую посещает агент, — вход, написанный не пользователем. Каждая форма на каждой странице — потенциальный командный канал. Корпус атак 2025–2026 показывает, что это не гипотеза: Tainted Memories позволяет атакующему привязать вредоносные инструкции к памяти агента через специально созданную страницу; HashJack прячет команды в URL fragments, которые посещает агент; hijacks Perplexity Comet срабатывают в один клик.

Защитная картина неудобна. Руководитель OpenAI preparedness сказал вслух тихую часть: indirect prompt injection «не является багом, который можно полностью исправить». Причина в том, что атака живет на границе чтения и действия агента, которая архитектурно размыта: каждый токен, который читает модель, в принципе может быть прочитан как инструкция.

Этот урок называет поверхность атаки, называет ландшафт бенчмарков (BrowseComp, OSWorld, WebArena-Verified) и моделирует минимальный сценарий indirect-prompt-injection, чтобы вы могли рассуждать о реальных защитах в Lessons 14 и 18.

## Концепция

### Ландшафт 2026 года, по одному абзацу на систему

**ChatGPT agent (OpenAI).** Запущен в июле 2025. Объединяет Operator (browsing) и Deep Research (multi-hour research). Standalone Operator был закрыт 31 августа 2025. SOTA на BrowseComp — 68.9%; сильные числа на OSWorld и WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Приобретение Vercept компанией Anthropic было сфокусировано на computer-use capabilities. Подняло Claude Sonnet на OSWorld с <15% до 72.5%. Claude Computer Use поставляется как tool API.

**Gemini 3 Pro with Browser Use (DeepMind).** Интеграция Browser Use поставляет computer-use controls; FSF v3 (апрель 2026, Lesson 20) отслеживает автономность конкретно в домене ML R&D.

**WebArena-Verified (ServiceNow, ICLR 2026).** Исправляет хорошо задокументированную проблему: исходный WebArena имел ~11.3% false-negative rate (задачи помечались как проваленные, хотя были решены). Релиз Verified переоценивает задачи с human-curated success criteria и добавляет Hard subset из 258 задач (статья ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp против OSWorld против WebArena

| Бенчмарк | Что измеряет | Горизонт |
|---|---|---|
| BrowseComp | Поиск конкретных фактов в открытом web под давлением времени | минуты |
| OSWorld | Агент управляет полным desktop (mouse, keyboard, shell) | десятки минут |
| WebArena-Verified | Transactional web tasks на симулированных сайтах | минуты |
| Hard subset | Задачи WebArena-Verified с multi-page state transitions | десятки минут |

Разные оси. Высокий результат BrowseComp говорит, что агент находит факты; он не говорит, что агент может забронировать рейс. Результат OSWorld ближе к «работает ли это на моем desktop». WebArena-Verified ближе к «может ли завершить flow». Любое production-решение требует бенчмарка, совпадающего с распределением задач.

### Поверхность атаки, по именам

1. **Indirect prompt injection.** Недоверенный контент страницы содержит инструкции. Агент читает их. Агент выполняет их. Публичные примеры: 2024 Kai Greshake et al., статья 2025 Tainted Memories, 2026 HashJack (Cato Networks).
2. **URL fragment / query injection.** `#fragment` или query string обходного URL содержит команды. Никогда не рендерится видимо; все равно находится внутри контекста агента.
3. **Memory-binding attacks.** Страница инструктирует агента записать persistent memory (Lesson 12 покрывает durable state). В следующей сессии память запускает payload без видимого trigger.
4. **CSRF-shaped attacks on authenticated sessions.** Класс Tainted Memories: агент залогинен где-то; страница атакующего выпускает state-changing requests, которые агент выполняет с cookies пользователя.
5. **One-click hijack.** Визуально безобидная кнопка несет последующий payload, которому следует агент. Класс Comet.
6. **Content-Security-Policy holes in the agent's host surface.** Rendering и tool layers сами могут быть attack vectors; стек browser-in-a-browser-agent широк.

### Почему «нельзя полностью исправить»

Атака изоморфна возможности агента. Агент должен читать недоверенный контент, чтобы выполнять работу. Любой контент, который читает агент, может содержать инструкции. Любые инструкции, которым следует агент, могут быть невыровнены с реальным запросом пользователя. Защиты (trust boundaries, classifiers, tool allowlists, HITL on consequential actions) повышают цену атаки и уменьшают ее blast radius. Они не закрывают класс.

Это тот же паттерн рассуждения, что и теорема Лёба (Lesson 8): агент не может доказать, что следующий токен безопасен; он может только построить систему, где небезопасные токены более обнаружимы.

### Защитная позиция, которая реально поставляется

- **Read / write boundary.** Чтение никогда не имеет последствий. Запись (submit form, posting content, вызов инструмента с side effects) требует свежего human approval, если инициирующий контент пришел извне trust boundary.
- **Tool allowlist per task.** Агент может browse; он не может инициировать wire transfer, если этот инструмент не был явно включен для задачи. Lesson 13 покрывает budgets.
- **Session isolation.** Сессии browser agent запускаются только со scoped credentials. Нет production auth, нет personal email. Логи каждого HTTP request сохраняются для аудита.
- **Content sanitizer.** Полученный HTML очищается от known-bad patterns перед конкатенацией в model context. (Снижает простые атаки; не останавливает sophisticated payloads.)
- **HITL on consequential actions.** Паттерн propose-then-commit (Lesson 15).
- **Canary tokens on memory.** Если memory entry срабатывает, пользователь это видит (Lesson 14).

## Использование

`code/main.py` моделирует крошечный запуск browser-agent против трех синтетических страниц. Одна страница benign, одна содержит direct prompt-injection blob в видимом тексте, одна содержит URL-fragment injection (не видимую, но внутри контекста агента). Скрипт показывает (a) что сделал бы naïve agent, (b) что ловит read/write boundary, (c) что ловит sanitizer, (d) что не ловит ни один.

## Результат

`outputs/skill-browser-agent-trust-boundary.md` задает scope предлагаемого развертывания browser-agent: какие trust zones оно затрагивает, что ему разрешено записывать и какие защиты должны быть на месте перед первым запуском.

## Упражнения

1. Запустите `code/main.py`. Определите, какую атаку ловит sanitizer, но не read/write boundary, и какую атаку ловит только read/write boundary.

2. Расширьте sanitizer, чтобы обнаруживать один класс HashJack-style URL-fragment injection. Измерьте false-positive rate на benign URLs с легитимными fragments.

3. Выберите один реальный workflow browser-agent, который вы знаете (например, «book a flight»). Перечислите каждое read и каждое write. Отметьте, какие writes требуют HITL и почему.

4. Прочитайте статью WebArena-Verified ICLR 2026. Определите одну категорию задач, где scoring исходного WebArena был ненадежным, и объясните, как subset Verified это исправляет.

5. Спроектируйте memory canary для настройки browser-agent. Что вы будете хранить, где, и что запускает alarm?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|---|---|---|
| Indirect prompt injection | «Плохой текст страницы» | Недоверенный контент на странице, которую читает агент, содержит инструкции, которые агент выполняет |
| Tainted Memories | «Атака на память» | Агент записывает attacker-supplied instruction в durable memory; срабатывает в следующей сессии |
| HashJack | «Атака через URL fragment» | Payload, скрытый в URL fragment / query string, находится в контексте агента, но не рендерится видимо |
| One-click hijack | «Плохая кнопка» | Видимый affordance несет follow-on payload, который агент выполняет |
| BrowseComp | «Web search benchmark» | Поиск конкретных фактов в open web; горизонт в масштабе минут |
| OSWorld | «Desktop benchmark» | Полный OS control; многошаговые GUI tasks |
| WebArena-Verified | «Исправленный web-task benchmark» | Переоцененный WebArena от ServiceNow с Hard subset |
| Read/write boundary | «Side-effect gate» | Чтение никогда не имеет последствий; запись требует свежего approval, если контент вне trust |

## Дополнительное чтение

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — объединение Operator и deep research; BrowseComp SOTA.
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — линия Operator и архитектура, ставшая ChatGPT agent.
- [Zhou et al. — WebArena](https://webarena.dev/) — исходный бенчмарк.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — статья ICLR 2026 о fixed-subset.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — включает обсуждение attack-surface для computer-use agents.
