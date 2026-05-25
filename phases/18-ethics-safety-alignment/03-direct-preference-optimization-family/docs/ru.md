# Семейство Direct Preference Optimization

> Rafailov et al. (2023) показали, что оптимум RLHF имеет замкнутую форму через данные предпочтений, поэтому можно пропустить явную модель вознаграждения и оптимизировать политику напрямую. Это наблюдение породило семейство - IPO, KTO, SimPO, ORPO, BPO, - где каждый метод исправляет отдельный режим отказа DPO. В 2026 году direct alignment algorithms используются в большем числе frontier post-training-запусков, чем PPO. Но over-optimization curve из Lesson 2 все еще применима: DAAs не избегают Goodhart, они лишь меняют место, где он проявляется.

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## Цели обучения

- Вывести замкнутую форму DPO из оптимума RLHF-with-KL.
- Сформулировать, какой режим отказа DPO исправляет каждый из IPO, KTO, SimPO, ORPO, BPO.
- Отличать "implicit reward gap" от "preference strength" и объяснить, почему identity mapping в IPO важен.
- Объяснить, почему Rafailov et al. (NeurIPS 2024) доказывают, что DAAs over-optimize, несмотря на отсутствие явной RM.

## Проблема

Целевая функция RLHF (Lesson 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

имеет известный оптимум:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Значит, вознаграждение неявно задается отношением оптимальной политики к reference:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Подставьте это в likelihood предпочтений Брэдли-Терри, и partition function `Z(x)` сократится, потому что зависит только от `x`. Останется loss только по параметрам политики - модель вознаграждения не нужна. Это и есть DPO.

Сложность в том, что вывод предполагает: оптимум достижим, данные предпочтений находятся in-distribution, а reference policy является истинным mode anchor. Ничего из этого не выполняется точно. Каждый член семейства исправляет свое нарушенное предположение.

## Концепция

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

Что может пойти не так:

- Implicit reward gap `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` неограничен. Крошечное предпочтение может породить произвольно большой разрыв.
- Loss толкает chosen и rejected log-probs в противоположные стороны. Он может снижать absolute log-prob chosen-ответа, пока rejected падает быстрее. Это феномен Degraded Chosen Response.
- Out-of-distribution preferences (редкая-редкая пара против редкой-редкой пары) порождают произвольные implicit rewards.

### IPO (Azar et al., 2024)

Identity Preference Optimization заменяет log-sigmoid на identity mapping по вероятности предпочтения. Loss становится squared-error по ограниченной цели:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Margin ограничен `1/(2 beta)`. Preference strength и implicit-reward gap пропорциональны. Раздувания нет.

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization полностью отказывается от попарной структуры. Для одного размеченного выхода и бинарного сигнала "desirable" или "undesirable" он отображает это в utility из prospect theory:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

с разными весами для gains и losses (loss aversion). Польза: можно использовать unpaired data, которых намного больше.

### SimPO (Meng et al., 2024)

Simple Preference Optimization выравнивает обучающий сигнал с генерацией. Уберите reference policy полностью и нормализуйте log-likelihood по длине:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

с margin `gamma` для стабилизации. Нормализация по длине убирает стимул эксплуатировать DPO length-bias failure mode (более длинный `y_w` по построению дает больший log-prob gap).

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization добавляет preference term к стандартному SFT negative log-likelihood:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Без reference policy - SFT-член является регуляризатором. Обучение в один этап от base model до aligned model. Отдельный SFT checkpoint не нужен.

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

Выявляет проблему Degraded Chosen Responses: DPO сохраняет ранжирование `y_w > y_l`, но absolute log-prob `y_w` может падать. BPO добавляет однострочную поправку, которая штрафует движения вниз для chosen response. Сообщается о +10.1% accuracy на Llama-3.1-8B-Instruct в math reasoning относительно DPO.

### Универсальный результат: DAAs все равно over-optimize

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) обучали политики с DPO, IPO, SLiC на нескольких датасетах и KL budgets. Кривые gold-reward-vs-KL имеют ту же форму peak-and-collapse, что у Gao et al. Implicit reward обращается к out-of-distribution samples во время обучения; KL-регуляризация это не стабилизирует.

DAAs не избегают Goodhart. Они меняют поверхность, где он проявляется, с "reward model over-optimized" на "reference policy ratio over-optimized." Универсальное исправление - лучшие данные, ансамбли, early stopping - применимо к обоим.

### Как выбирать между ними (2026)

- Если у вас много paired preference data: DPO с консервативным beta, SimPO, если виден length bias.
- Если у вас unpaired binary feedback: KTO.
- Если вам нужен single-stage pipeline от base model: ORPO.
- Если в DPO logs видны degraded chosen log-probs: BPO.
- Если preference strengths широко варьируют и DPO насыщается: IPO.

Каждая лаборатория запускает все пять на battery и выбирает победителя под задачу. Нет причины ожидать, что optimum одинаков для math reasoning и safety.

## Использование

`code/main.py` сравнивает шесть losses (DPO, IPO, KTO, SimPO, ORPO, BPO) на игрушечном датасете предпочтений, где истинная preference strength меняется от пары к паре. Каждый loss оптимизируется на одной и той же выборке из 500 пар с маленькой softmax policy. Строятся final win rate, chosen-log-prob drift и implicit-reward spread по методам.

## Результат

Этот урок создает `outputs/skill-preference-loss-selector.md`. По статистикам датасета (paired vs unpaired, variable vs uniform preference strength, length distribution) и цели (single-stage или SFT-then-preference) он рекомендует preference loss и сообщает режим отказа, от которого тот защищает.

## Упражнения

1. Запустите `code/main.py`. Сообщите final chosen-log-prob drop для DPO и BPO. BPO должен сохранять более высокую absolute probability для chosen - проверьте это.

2. Модифицируйте данные предпочтений так, чтобы все пары имели одинаковую strength. Какой из шести методов наиболее устойчив? Какой деградирует? Объясните преимущество IPO здесь.

3. Сделайте rejected responses в среднем в 2x длиннее, чем chosen. Не меняя ничего больше, численно покажите length exploitation у DPO и исправление SimPO.

4. Rafailov et al. (NeurIPS 2024) утверждают, что DAAs over-optimize. Воспроизведите single-point version: постройте chosen-minus-rejected KL divergence и наблюдайте over-optimization в DPO при большом beta.

5. Прочитайте аннотацию статьи BPO (OpenReview b97EwMUWu7). Запишите однострочную поправку, которую BPO добавляет к DPO. Сверьте с реализацией в `code/main.py`.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| DPO | "RLHF without a reward model" | Loss, выведенный из замкнутого оптимума RLHF; только параметры политики |
| Implicit reward | "the log-ratio" | `beta * log(pi(y|x) / pi_ref(y|x))` - вознаграждение, подразумеваемое DPO |
| IPO | "bounded DPO" | Заменяет log-sigmoid на identity; implicit reward gap ограничен `1/(2 beta)` |
| KTO | "unpaired DPO" | Prospect-theory utility по одиночным меткам с loss aversion |
| SimPO | "reference-free DPO" | Length-normalized log-likelihood + margin; без reference policy |
| ORPO | "one-stage DPO" | NLL + odds-ratio preference term; обучает base model за один проход |
| BPO | "chosen-preserving DPO" | DPO плюс штраф за снижение absolute log-prob chosen response |
| Degraded Chosen | "chosen goes down" | DPO снижает chosen log-prob, пока rejected падает быстрее |
| DAA | "direct alignment algorithm" | Любой preference-loss метод, который пропускает явную RM |

## Дополнительное чтение

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
