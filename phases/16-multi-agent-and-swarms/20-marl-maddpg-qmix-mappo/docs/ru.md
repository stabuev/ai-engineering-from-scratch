# MARL — MADDPG, QMIX, MAPPO

> Наследие reinforcement learning в многоагентной координации, которое все еще влияет на LLM-agent systems в 2026 году. **MADDPG** (Lowe et al., NeurIPS 2017, arXiv:1706.02275) ввел Centralized Training, Decentralized Execution (CTDE): каждый critic видит состояния и действия всех agents во время training; во время test запускаются только local actors. Работает для cooperative, competitive и mixed settings. **QMIX** (Rashid et al., ICML 2018, arXiv:1803.11485) — value-decomposition с monotonic mixing network; per-agent Qs объединяются в joint Q так, что `argmax` аккуратно распределяется — доминирующий подход на StarCraft Multi-Agent Challenge (SMAC). **MAPPO** (Yu et al., NeurIPS 2022, arXiv:2103.01955) — PPO с centralized value function; "surprisingly effective" на particle-world, SMAC, Google Research Football, Hanabi с минимальной настройкой. Эти идеи лежат в основе training policies для agent teams, которые должны действовать децентрализованно. MAPPO — **default 2026 cooperative-MARL baseline**. В этом уроке каждая идея строится на маленькой grid-world toy, чтобы CTDE, value decomposition и centralized critics стали привычными до перехода к LLM-agent training.

**Тип:** Изучение
**Языки:** Python (stdlib, small NumPy-free implementations)
**Требования:** Фаза 09 (Reinforcement Learning), Фаза 16 · 09 (Parallel Swarm Networks)
**Время:** ~90 минут

## Цели обучения

- Объяснять наследие MARL (MADDPG, QMIX, MAPPO) за мультиагентной координацией.
- Различать CTDE, декомпозицию ценности и дефолт MAPPO.
- Говорить, почему инженерам LLM-агентов важен классический MARL.

## Задача

LLM-agent systems все чаще обучают policies для межагентной координации: когда уступить, когда действовать, к какому peer обратиться. Литература, которая объясняет, как обучать такие policies, — это Multi-Agent Reinforcement Learning (MARL), возникшая до волны LLM и имеющая небольшой набор доминирующих алгоритмов.

Читать статьи по MARL без словаря паттернов болезненно. Centralized training with decentralized execution (CTDE), value decomposition и centralized critics — не модные слова, а конкретные ответы на конкретные проблемы:

- Independent RL (каждый agent учится один) нестационарен с точки зрения каждого agent. Плохо.
- Centralized RL (один agent управляет всем) не масштабируется и нарушает execution constraints.
- CTDE берет лучшее из обоих: обучение с global information, deployment с local policies.

## Концепция

### Три среды, которые используют статьи

- **Particle World (multi-agent particle env).** Простая 2D physics с cooperative/competitive tasks. Исходный testbed MADDPG.
- **StarCraft Multi-Agent Challenge (SMAC).** Cooperative micro-management, partial observation. Testbed QMIX. Discrete actions, continuous states.
- **Google Research Football, Hanabi, MPE.** Baselines MAPPO.

У разных envs разные action/observation types. Алгоритмы выбираются соответственно.

### MADDPG (2017) — паттерн CTDE

У каждого agent `i` есть actor `mu_i(o_i)`, который отображает собственное observation в action. У каждого agent также есть critic `Q_i(x, a_1, ..., a_n)`, который видит все observations и все actions во время training. Actor обновляется policy gradient относительно оценки critic.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

Почему CTDE: во время training мы знаем действия всех; используем это, чтобы уменьшить variance в каждом critic. Во время deploy каждый agent видит только `o_i` и вызывает `mu_i(o_i)`.

Режим отказа: critics растут с N agents (input включает все actions). Без аппроксимаций плохо масштабируется дальше ~10 agents.

### QMIX (2018) — value decomposition

Только cooperative. Global reward — сумма монотонной функции per-agent Q-values:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

Монотонность гарантирует, что `argmax_a Q_tot` можно вычислить, когда каждый agent независимо выбирает `argmax_{a_i} Q_i`. Это **именно то свойство decentralized execution**, которое вам нужно. Во время training mixing network производит `Q_tot` из per-agent Qs.

Почему QMIX выигрывает на SMAC: cooperative StarCraft micro-management имеет homogeneous agents, local obs, global reward — идеальное соответствие для value decomposition.

Режим отказа: ограничение монотонности жесткое; некоторые задачи имеют reward structures, которые не являются monotone decomposable (один agent жертвует собой ради команды). Расширения (QTRAN, QPLEX) ослабляют это.

### MAPPO (2022) — недооцененный дефолт

Multi-Agent PPO: PPO с centralized value function. У каждого agent своя policy; все agents разделяют (или имеют per-agent) value functions, которые видят full state. Yu et al. 2022 сравнили MAPPO с MADDPG, QMIX и их расширениями на пяти benchmarks и обнаружили:

- MAPPO не уступает или превосходит off-policy MARL methods на particle-world, SMAC, Google Research Football, Hanabi, MPE.
- Требует минимальной hyperparameter tuning.
- Стабильное training; воспроизводимо по seeds.

Сообщество недооценивало on-policy MARL до этой статьи. В 2026 году MAPPO — default baseline для cooperative MARL; любой новый method должен его превзойти.

### Почему LLM-agent engineers должны об этом заботиться

Три прямых применения:

1. **Router training.** Meta-agent выбирает, какой sub-agent обработает task. Это MARL problem с N decentralized sub-agents и одним centralized router. MAPPO подходит.
2. **Role emergence.** В generative-agent simulations обучение agents принимать взаимодополняющие роли со временем — скрытая MARL problem. QMIX-style value decomposition принудительно задает complementarity конструкцией.
3. **Multi-agent tool use.** Когда agents делят tools и конкурируют за budget, обучение через CTDE дает deployable local policies, которые соблюдают resource constraints.

Практическая оговорка: в 2026 году большинство production LLM-agent systems промптят свои policies, а не обучают их. MARL нужен, когда у вас есть (a) много interaction data, (b) четкий reward signal и (c) готовность инвестировать в training infrastructure.

### CTDE как design pattern за пределами RL

Даже без training CTDE полезен как архитектурный паттерн:

- Во время *design* предполагайте полную видимость команды.
- Во время *runtime* обеспечивайте decentralized execution: каждый agent видит только `o_i`.

Паттерн заставляет явно держать per-agent state и заранее думать о partial observability. Многие production multi-agent systems молча предполагают общий state везде — дисциплина CTDE предотвращает это.

### Проблема нестационарности

Когда несколько agents учатся одновременно, environment каждого agent (включая policies других) нестационарна. Классические доказательства single-agent RL ломаются. Все MARL algorithms в этом уроке решают это:

- MADDPG: global critic видит все actions, поэтому его value estimate стационарнее.
- QMIX: value decomposition переносит learning в joint-Q space, где optimality хорошо определена.
- MAPPO: centralized value function сглаживает variance от изменений policies других.

В LLM-agent systems нестационарность проявляется как "мой agent работал в прошлом месяце, теперь upstream agent изменился, и мой ведет себя неправильно." Training MARL with CTDE — принципиальное исправление; prompt-level fixes быстрее, но менее долговечны.

### Что этот урок НЕ покрывает

Training actual networks — тема Фазы 09. Этот урок строит scripted-policy versions, которые демонстрируют CTDE, value-decomposition и centralized-value patterns без gradient updates. Цель — усвоить паттерны до выбора полноценной MARL library (PyMARL, MARLlib, RLlib multi-agent).

## Сборка

`code/main.py` реализует три демонстрации паттернов, все на tiny 2-agent cooperative grid-world:

- Environment: 2 agents на 4x4 grid, одна reward pellet. Reward = 1, если любой agent достигает pellet; task завершается.
- `IndependentAgents` — каждый agent считает других частью environment. Baseline.
- `MADDPGStyle` — centralized critic вычисляет joint value; actor policies обновляются от него. Scripted policy improvement.
- `QMIXStyle` — value decomposition с monotone mixer.
- `MAPPOStyle` — centralized value function; policies обновляются относительно shared baseline.

Все четыре запускают одинаковые episodes и сообщают average steps-to-goal. Варианты CTDE сходятся к более коротким paths, чем independent baseline.

Запуск:

```
python3 code/main.py
```

Ожидаемый вывод: independent agents в среднем требуют ~6 steps; CTDE variants сходятся к ~3.5 steps (optimal для 4x4 grid — 3). Разница паттернов проявляется несмотря на scripted policies.

## Использование

`outputs/skill-marl-picker.md` — skill, который выбирает MARL algorithm для заданной multi-agent task: cooperative vs competitive, homogeneous vs heterogeneous, action-space type, scale, reward signal.

## Доставка

MARL в production встречается редко. Когда вы его используете:

- **Начните с MAPPO.** Статья 2022 года установила его как baseline; сначала воспроизвести его — значит сэкономить недели охоты за более сложными methods.
- **Логируйте stream наблюдений и действий каждого agent.** Отладка MARL без per-agent traces безнадежна.
- **Отделяйте training code от execution code.** CTDE — дисциплина; пусть execution path действительно видит только `o_i`.
- **Предупреждение о reward shaping.** MARL исключительно чувствителен к reward design. Один coordination bug в shaping, и agents научатся его эксплуатировать. Запускайте adversarial tests.
- **Для LLM agents** сначала рассмотрите prompt-level policies. Инвестируйте в MARL training только когда interaction data + reward signal + infrastructure уже есть.

## Упражнения

1. Запустите `code/main.py`. Измерьте разрыв steps-to-goal между independent и MAPPO-style agents. Растет или уменьшается разрыв на 6x6 grid?
2. Реализуйте competitive variant: два agents, одна pellet, reward получает только первый достигший. Какой pattern чисто обрабатывает competition? Исторически MADDPG.
3. Прочитайте MADDPG (arXiv:1706.02275) Section 3. Реализуйте точное critic update rule символически в pseudocode своими словами.
4. Прочитайте MAPPO (arXiv:2103.01955). Почему авторы утверждают, что centralized value + PPO превосходит off-policy MARL на их benchmarks? Перечислите три сильнейших claims.
5. Примените CTDE как design pattern к гипотетической LLM-agent system (например, research agent + summarizer + coder). Какая joint information доступна во время design, но недоступна во время runtime?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| MARL | "Multi-Agent RL" | Reinforcement learning для многоагентных систем. |
| CTDE | "Centralized Training, Decentralized Execution" | Обучение с global info; deployment с local policies. |
| MADDPG | "Multi-Agent DDPG" | CTDE с per-agent critic, который видит все observations + actions. |
| QMIX | "Value decomposition" | Монотонное смешивание per-agent Qs. Cooperative. |
| MAPPO | "Multi-Agent PPO" | PPO с centralized value function. Дефолтный baseline 2026 года. |
| Value decomposition | "Сумма individual Qs" | Joint Q представлен как monotone function от per-agent Qs. |
| Non-stationarity | "Движущиеся цели" | Env каждого agent меняется, пока другие учатся. Центральная проблема MARL. |
| On-policy / off-policy | "Учиться на current / replay" | PPO является on-policy (MAPPO); DDPG и Q-learning являются off-policy. |
| SMAC | "StarCraft Multi-Agent Challenge" | Cooperative micromanagement benchmark; родная площадка QMIX. |

## Дополнительное чтение

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO; NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — readable framing результата MAPPO
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
