# Sim-to-Real Transfer

> Политика, обученная в симуляторе и проваливающаяся на железе, — это политика, которая запомнила симулятор. Domain randomization, domain adaptation и system identification — три инструмента, которые помогают обученным контроллерам преодолеть reality gap.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 9 · 08 (PPO), Фаза 2 · 10 (Bias/Variance)
**Время:** ~45 минут

## Цели обучения

- Объяснять reality gap и почему политика, обученная только в симуляторе, может его запомнить.
- Обучать политику с domain randomization и оценивать ее zero-shot перенос.
- Различать domain randomization, domain adaptation и system identification.

## Проблема

Обучать реального робота медленно, опасно и дорого. Двуногому роботу нужны миллионы обучающих эпизодов, чтобы научиться ходить; реальный двуногий робот, упавший хотя бы один раз, ломает железо. Симуляция дает неограниченные reset-ы, детерминированную воспроизводимость, параллельные среды и отсутствие физического ущерба.

Но симуляторы ошибаются. В подшипниках больше трения, чем в моделях MuJoCo. У камер есть lens distortion, которого нет в симуляторе. У моторов есть задержки, backlash и saturation, которые пропускают 99% sim-моделей. Ветер, пыль и переменное освещение ломают политику, обученную на стерильном rendering. **Reality gap** — систематическое различие между sim distribution и real distribution — центральная проблема deployed RL для робототехники.

Вам нужна политика, *устойчивая к sim-to-real distribution shift*. Три исторических подхода: рандомизировать симулятор (domain randomization), адаптировать политику с небольшим количеством реальных данных (domain adaptation / fine-tuning) или идентифицировать параметры реальной системы и подогнать их (system identification). В 2026 году доминирующий рецепт объединяет все три подхода с массовой параллельной симуляцией (Isaac Sim, Isaac Lab, Mujoco MJX на GPU).

## Концепция

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018. Во время обучения рандомизируйте каждый sim-параметр, который может отличаться на реальном роботе: массы, коэффициенты трения, motor PD gains, шум сенсоров, положение камеры, освещение, текстуры, contact models. Политика учит условное распределение по тому, "в каком sim она сегодня находится", и обобщает по всему диапазону. Если реальный робот попадает внутрь training envelope, политика работает.

- **Плюс:** реальные данные не нужны. Один рецепт, много роботов.
- **Минус:** чрезмерно рандомизированное обучение дает "универсальную", но слишком осторожную политику. Слишком много шума ≈ слишком много regularization.

**System Identification (SI).** Подгоните параметры симулятора к real-world данным до обучения. Если вы можете измерить трение в шарнире руки на реальном роботе, подставьте его в sim. Затем обучайте политику, которая ожидает эти значения. Нужен доступ к реальной системе, зато reality gap уменьшается напрямую.

- **Плюс:** точная обучающая цель с низким уровнем шума.
- **Минус:** остаточная ошибка модели невидима для политики; небольшие неидентифицированные эффекты (например, motor deadband) все еще ломают deployment.

**Domain Adaptation.** Обучите в sim, затем fine-tune с небольшим количеством реальных данных. Два варианта:

- **Real2Sim2Real:** обучите residual simulator `f(s, a, z) - f_sim(s, a)` на реальных rollouts, затем обучайте в исправленном sim. Закрывает gap без большого объема реальных данных.
- **Observation adaptation:** обучите политику, которая отображает real obs → sim-like obs через обученный feature extractor (например, GAN pixel-to-pixel). Контроллер остается в sim.

**Privileged learning / teacher-student.** Miki et al. 2022 (ANYmal quadruped). Обучите *teacher* в симуляции с доступом к privileged information (ground truth friction, terrain height, IMU drift). Дистиллируйте *student*, который видит только наблюдения реальных сенсоров. Student учится выводить privileged features из истории и остается устойчивым к физическим параметрам.

**Massively parallel simulation.** 2024–2026. Isaac Lab, Mujoco MJX, Brax запускают тысячи параллельных роботов на одном GPU. PPO с 4,096 параллельными humanoids собирает годы опыта за часы. "Reality gap" сжимается по мере расширения training distribution; DR становится почти бесплатной, когда у каждой из этих 4,096 envs свои рандомизированные параметры.

**Реальный рецепт 2026 года (пример ходьбы quadruped):**

1. Massively parallel sim с domain-randomized gravity, friction, motor gains, payload.
2. Teacher policy, обученная с privileged info (terrain map, body velocity ground truth).
3. Student policy, дистиллированная из teacher только по proprioception (leg joint encoders).
4. Опциональная observation adaptation через autoencoder на real IMU.
5. Deploy. Zero-shot на 10+ средах. Если не сработало, сделайте минуты real-world fine-tuning с safety-constrained PPO.

## Постройте это

Код этого урока — маленькая демонстрация domain randomization в GridWorld с *шумными* переходами. Мы обучаем политику, которая сталкивается с рандомизированными вероятностями slip в "sim", и оцениваем ее в "real" с уровнем slip, которого она никогда не видела во время обучения. Эта форма напрямую соответствует переносу MuJoCo-to-hardware.

### Шаг 1: параметризованный sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` — параметр, который предоставляет симулятор. В реальной робототехнике это может быть friction, mass, motor gain — что угодно, что сдвигается между sim и real.

### Шаг 2: обучение с DR

В начале каждого эпизода сэмплируйте `slip ~ Uniform[0.0, 0.4]`. Обучайте PPO / Q-learning / что угодно. Делайте это много эпизодов.

### Шаг 3: zero-shot оценка на "real" slips

Оценивайте на `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Первые четыре находятся внутри training support; `0.5` и `0.7` — вне его. DR-trained policy должна оставаться почти оптимальной внутри support и деградировать плавно за его пределами. Fixed-slip-trained policy будет хрупкой вне своего training slip.

### Шаг 4: сравнение с узким обучением

Обучите вторую политику только с `slip = 0.0`. Оцените на том же sweep по `slip`. Вы должны увидеть катастрофическое падение сразу после того, как real slip > 0.

## Подводные камни

- **Слишком много randomization.** Обучайте на `slip ∈ [0, 0.9]`, и ваша политика станет настолько risk-averse, что никогда не попробует оптимальный путь. Подгоняйте *ожидаемое* real-world распределение, а не "может случиться что угодно".
- **Слишком мало randomization.** Обучение на тонком срезе вообще не дает политике обобщать. Используйте adaptive curriculum (Automatic Domain Randomization), который расширяет распределение по мере улучшения политики.
- **Неверно идентифицированное пространство параметров.** Рандомизируете не то (camera hue, когда реальный gap — motor delay), и DR не помогает. Сначала профилируйте реального робота.
- **Утечка privileged info.** Teacher, который использует global state для действий, а не только observations, может породить student, который не сможет догнать. Убедитесь, что policy teacher реализуема student-ом по истории наблюдений.
- **Провал sim-to-sim transfer.** Если ваша политика не устойчива к более сложному варианту sim, она не будет устойчивой и к реальному миру. Всегда тестируйте на held-out sim variant перед deployment.
- **Нет real-world safety envelope.** Политика, которая работает в sim и "работает в real" без low-level safety shield, все равно может сломать железо. Добавьте rate limits, torque limits, joint limits в не-обучаемый controller.

## Используйте это

Sim-to-real stack 2026 года:

| Домен | Stack |
|--------|-------|
| Legged locomotion (ANYmal, Spot, humanoid) | Isaac Lab + DR + privileged teacher / student |
| Manipulation (dexterous hands, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl (DR at unprecedented scale) |
| Industrial arms | MuJoCo-Warp + SI + small real fine-tune |

Для control на любых масштабах workflow одинаков: подгоните sim как можно лучше, рандомизируйте то, что не можете подогнать, обучите огромные политики, дистиллируйте, deploy с safety shield.

## Результат

Сохраните как `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Упражнения

1. **Легко.** Обучите Q-learning agent на fixed-slip GridWorld (slip=0.0). Оцените на slip ∈ {0.0, 0.1, 0.3, 0.5}. Постройте график return vs slip.
2. **Средне.** Обучите DR Q-learning agent, сэмплирующий `slip ~ Uniform[0, 0.3]`. Оцените тот же sweep. Сколько DR дает на slip=0.5 (out-of-distribution)?
3. **Сложно.** Реализуйте curriculum: начните со slip=0.0, расширяйте диапазон DR каждый раз, когда policy достигает 90% от optimal. Измерьте общее число environment steps, чтобы достичь slip=0.3 zero-shot, и сравните с fixed DR baseline.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Reality gap | "Sim-to-real difference" | Distribution shift между физикой/сенсорами в training и deployment. |
| Domain randomization (DR) | "Train across random sims" | Рандомизировать sim parameters во время обучения, чтобы policy обобщала. |
| System identification (SI) | "Measure real and fit sim" | Оценить реальные физические параметры; настроить sim под них. |
| Domain adaptation | "Fine-tune on real data" | Небольшой real-world fine-tune после sim training; может адаптировать obs или dynamics. |
| Privileged info | "Ground truth for teacher" | Информация, которая есть только в sim; student должен вывести ее из obs history. |
| Teacher/student | "Distill privileged -> observable" | Teacher обучен с shortcuts; student учится имитировать без них. |
| ADR | "Automatic Domain Randomization" | Curriculum, который расширяет DR ranges по мере улучшения policy. |
| Real2Sim | "Close the gap with real data" | Обучить residual, чтобы sim имитировал real rollouts. |

## Дополнительное чтение

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — оригинальная статья о DR (vision for robotics).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR для dynamics, quadruped locomotion.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR at scale.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — teacher-student для ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — massively parallel sim, который движет deployment-ами 2025–2026.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — метод ADR curriculum.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna framing (использование model для planning + rollouts), лежащий в основе современных sim-to-real pipelines.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — taxonomy методов sim-to-real с benchmark results.
