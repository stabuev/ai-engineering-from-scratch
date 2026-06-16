# Статистика для машинного обучения

> Статистика помогает понять, действительно ли ваша модель работает или ей просто повезло.

**Тип:** Практика
**Язык:** Python
**Предварительные требования:** Фаза 1, уроки 06 (вероятность и распределения), 07 (теорема Байеса)
**Время:** ~120 минут

## Цели обучения

- Вычислять описательные статистики, корреляцию Pearson/Spearman и ковариационные матрицы с нуля
- Выполнять проверки гипотез (t-test, chi-squared) и правильно интерпретировать p-values и доверительные интервалы
- Использовать bootstrap-ресэмплинг для построения доверительных интервалов для любой метрики без предположений о распределении
- Отличать статистическую значимость от практической значимости с помощью мер размера эффекта

## Проблема

Вы обучили две модели. Model A получает 0.87 на тестовом наборе. Model B получает 0.89. Вы выкатываете Model B. Через три недели производственные метрики хуже, чем раньше. Что произошло?

Model B на самом деле не превзошла Model A. Разница 0.02 была шумом. Ваш тестовый набор был слишком маленьким, или дисперсия была слишком высокой, или и то и другое. Вы отправили в продакшен случайность, замаскированную под улучшение.

Такое происходит постоянно. Перестановки в лидбордах Kaggle. Статьи, результаты которых не удается воспроизвести. A/B-тесты, объявляющие победителей на нескольких сотнях примеров. Первопричина всегда одна: кто-то пропустил статистику.

Статистика дает инструменты, чтобы отличать сигнал от шума. Она говорит, когда различие реально, насколько вы можете быть уверены и сколько данных нужно, прежде чем результату можно доверять. Каждому ML-пайплайну, каждому сравнению моделей, каждому эксперименту нужна статистика. Без нее вы гадаете.

## Концепция

### Описательная статистика: краткое описание данных

Прежде чем строить модель, нужно понять, как выглядят ваши данные. Описательная статистика сжимает датасет до нескольких чисел, которые описывают его форму.

**Меры центральной тенденции** отвечают на вопрос «где середина?»

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Среднее — это точка баланса. Медиана — отметка середины. Когда они расходятся, распределение скошено. У распределений дохода среднее >> медианы (правый хвост из-за миллиардеров). У распределений loss во время обучения часто среднее << медианы (левый хвост из-за легких примеров).

**Меры разброса** отвечают на вопрос «насколько данные рассеяны?»

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Перцентили** делят отсортированные данные на 100 равных частей. 25-й перцентиль (Q1) означает, что 25% значений находятся ниже этой точки. 50-й перцентиль — это медиана. 75-й перцентиль — это Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

В ML перцентили важны для latency инференса, распределений уверенности предсказаний и понимания распределений ошибок. Модель с низкой средней ошибкой, но ужасной P99-ошибкой может быть бесполезной для safety-critical приложений.

**Выборочные и генеральные статистики.** Когда вы вычисляете дисперсию по выборке, делите на (n-1), а не на n. Это поправка Бесселя. Она компенсирует то, что выборочное среднее не равно истинному среднему генеральной совокупности. При n в знаменателе вы систематически занижаете истинную дисперсию. При (n-1) оценка несмещенная.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

На практике: если n велико (тысячи примеров), разница пренебрежимо мала. Если n мало (десятки примеров), это важно.

### Корреляция: как переменные движутся вместе

Корреляция измеряет силу и направление линейной связи между двумя переменными.

**Коэффициент корреляции Pearson** измеряет линейную ассоциацию:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson предполагает, что связь линейна, а обе переменные примерно нормально распределены. Он чувствителен к выбросам. Одна экстремальная точка может перетащить r с 0.1 до 0.9.

**Ранговая корреляция Spearman** измеряет монотонную ассоциацию:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Когда что использовать:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Золотое правило:** корреляция не означает причинность. Продажи мороженого и случаи утопления коррелируют, потому что оба показателя растут летом. Точность вашей модели и число параметров коррелируют, но добавление параметров не улучшает accuracy автоматически (см.: overfitting).

### Ковариационная матрица

Ковариация между двумя переменными измеряет, как они изменяются вместе:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Для d признаков ковариационная матрица C — это матрица d x d, где C[i][j] = Cov(feature_i, feature_j). Диагональные элементы C[i][i] — это дисперсии каждого признака.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Связь с PCA.** PCA выполняет eigendecomposition ковариационной матрицы. Собственные векторы — это главные компоненты (направления максимальной дисперсии). Собственные значения показывают, сколько дисперсии захватывает каждая компонента. Именно это разбиралось в уроке 10, но теперь видно, почему ковариационная матрица — правильный объект для разложения: она кодирует все попарные линейные отношения в данных.

**Связь с корреляцией.** Корреляционная матрица — это ковариационная матрица стандартизованных переменных (каждая разделена на свое стандартное отклонение). Корреляция нормализует ковариацию, так что все значения попадают в [-1, 1].

### Проверка гипотез

Проверка гипотез — это фреймворк для принятия решений в условиях неопределенности. Вы начинаете с утверждения, собираете данные и определяете, согласуются ли данные с этим утверждением.

**Постановка:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-value** — это вероятность увидеть данные настолько же экстремальные, как наблюдаемые, при условии, что H0 истинна. Это НЕ вероятность того, что H0 истинна. Это самое распространенное недопонимание в статистике.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Доверительные интервалы** дают диапазон правдоподобных значений параметра:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Ширина доверительного интервала говорит о точности. Широкие интервалы означают высокую неопределенность. Узкие интервалы означают, что оценка точна (но не обязательно верна, если данные смещены).

### t-test

t-test сравнивает средние. Есть несколько вариантов.

**Одновыборочный t-test:** отличается ли среднее генеральной совокупности от предполагаемого значения?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Двухвыборочный t-test (независимый):** различаются ли средние двух групп?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Парный t-test:** когда измерения идут парами (одна и та же модель оценивается на одних и тех же разбиениях данных):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

В ML парный t-test встречается часто: вы запускаете обе модели на одних и тех же 10 фолдах cross-validation и сравниваете их scores попарно.

### Chi-squared Test

Chi-squared test проверяет, совпадают ли наблюдаемые частоты с ожидаемыми. Полезно для категориальных данных.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B-тестирование ML-моделей

A/B-тестирование в ML — не то же самое, что веб A/B-тестирование. У сравнения моделей есть специфические сложности:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Процедура:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Статистическая значимость против практической значимости

Результат может быть статистически значимым, но практически бессмысленным. При достаточном количестве данных даже тривиальная разница становится статистически значимой.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Размер эффекта** количественно показывает, насколько велика разница, независимо от размера выборки:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Всегда сообщайте и p-value, и размер эффекта. p-value говорит, реальна ли разница. Размер эффекта говорит, имеет ли она значение.

### Проблема множественных сравнений

Когда вы проверяете много гипотез, некоторые окажутся «значимыми» случайно. Если проверить 20 вещей при alpha = 0.05, вы ожидаете 1 ложноположительный результат даже тогда, когда ничего реального нет.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Поправка Bonferroni:** разделите alpha на число тестов.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

В ML это важно, когда вы сравниваете модель по нескольким метрикам, тестируете много конфигураций гиперпараметров или оцениваете на нескольких датасетах.

### Bootstrap-методы

Bootstrap оценивает выборочное распределение статистики через ресэмплинг данных с возвращением. Предположения о базовом распределении не требуются.

**Алгоритм:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Доверительный интервал bootstrap (percentile method):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Почему bootstrap важен для ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap для сравнения моделей:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Это устойчивее, чем парный t-test, потому что не делает предположений о распределении.

### Параметрические и непараметрические тесты

**Параметрические тесты** предполагают конкретное распределение (обычно нормальное):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Непараметрические тесты** не делают предположений о распределении:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Когда использовать непараметрические тесты:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Когда использовать параметрические тесты:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

В ML-экспериментах обычно мало n (5 или 10 фолдов cross-validation), поэтому непараметрические тесты вроде Wilcoxon signed-rank часто уместнее, чем t-tests.

### Центральная предельная теорема: практические следствия

CLT говорит, что распределение выборочных средних приближается к нормальному распределению по мере роста n, независимо от распределения исходной генеральной совокупности.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Почему это важно для ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Чего CLT НЕ делает:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Распространенные статистические ошибки в ML-статьях

1. **Тестирование на обучающем наборе.** Гарантирует overfitting. Всегда откладывайте данные, которые модель никогда не видит во время обучения.

2. **Нет доверительных интервалов.** Одна accuracy без неопределенности делает результаты невоспроизводимыми и непроверяемыми.

3. **Игнорирование множественных сравнений.** Тестирование 50 конфигураций и публикация лучшей без поправки завышает долю ложноположительных результатов.

4. **Смешение статистической и практической значимости.** p-value 0.001 для улучшения accuracy на 0.01% не имеет смысла.

5. **Использование accuracy на несбалансированных данных.** 99% accuracy на датасете с 99% отрицательного класса означает, что модель ничему не научилась. Используйте precision, recall, F1 или AUC.

6. **Cherry-picking метрик.** Публикуется только метрика, где ваша модель выигрывает. Честная оценка показывает все релевантные метрики.

7. **Утечка информации между train/test split.** Нормализация до разбиения или использование будущих данных для предсказания прошлого.

8. **Маленькие тестовые наборы без оценок дисперсии.** Оценивать на 100 примерах и заявлять улучшение 2% — это шум, а не сигнал.

9. **Предположение независимости, когда данные зависимы.** Медицинские изображения одного пациента, несколько предложений из одного документа. Наблюдения внутри группы коррелированы.

10. **P-hacking.** Пробовать разные тесты, подмножества или критерии исключения, пока не получится p < 0.05. Результат становится артефактом поиска.

## Сборка

Вы реализуете:

1. **Описательные статистики с нуля** (mean, median, mode, standard deviation, percentiles, IQR)
2. **Функции корреляции** (Pearson и Spearman, вместе с ковариационной матрицей)
3. **Проверки гипотез** (one-sample t-test, two-sample t-test, chi-squared test)
4. **Bootstrap-доверительные интервалы** (для любой статистики, без предположений)
5. **Симулятор A/B-теста** (генерация данных, тестирование, проверка Type I и Type II errors)
6. **Демо статистической и практической значимости** (показывает, что большое n делает все «значимым»)

Все с нуля, используя только `math` и `random`. Без numpy, без scipy.

## Ключевые термины

| Термин | Определение |
|---|---|
| Mean | Сумма значений, деленная на их количество. Чувствителен к выбросам. |
| Median | Среднее значение отсортированных данных. Устойчива к выбросам. |
| Standard deviation | Квадратный корень из дисперсии. Измеряет разброс в исходных единицах. |
| Percentile | Значение, ниже которого находится заданный процент данных. |
| IQR | Межквартильный размах. Q3 минус Q1. Разброс средних 50%. |
| Pearson correlation | Измеряет линейную ассоциацию между двумя переменными. Диапазон [-1, 1]. |
| Spearman correlation | Измеряет монотонную ассоциацию с использованием рангов. |
| Covariance matrix | Матрица попарных ковариаций между всеми признаками. |
| Null hypothesis | Базовое предположение об отсутствии эффекта или различия. |
| p-value | Вероятность получить настолько экстремальные данные при истинной нулевой гипотезе. |
| Confidence interval | Диапазон правдоподобных значений параметра при заданном уровне доверия. |
| t-test | Проверяет, различаются ли средние значимо. Использует t-distribution. |
| Chi-squared test | Проверяет, отличаются ли наблюдаемые частоты от ожидаемых. |
| Effect size | Величина различия, независимая от размера выборки. Часто используют Cohen's d. |
| Bonferroni correction | Делит порог значимости на число тестов, чтобы контролировать ложноположительные результаты. |
| Bootstrap | Ресэмплинг с возвращением для оценки выборочных распределений. |
| Type I error | Ложноположительный результат. Отклонение H0, когда она истинна. |
| Type II error | Ложноотрицательный результат. Неспособность отклонить H0, когда она ложна. |
| Statistical power | Вероятность правильно отклонить ложную H0. Power = 1 минус частота Type II error. |
| Central limit theorem | Выборочные средние сходятся к нормальному распределению по мере роста размера выборки. |
| Parametric test | Предполагает конкретное распределение данных (обычно нормальное). |
| Non-parametric test | Не делает предположений о распределении. Работает с рангами или знаками. |

## Дополнительное чтение

- [Seeing Theory (Университет Брауна)](https://seeing-theory.brown.edu/) — интерактивное визуальное введение в вероятность и статистику.
- [Khan Academy — Statistics and Probability](https://www.khanacademy.org/math/statistics-probability) — полный бесплатный курс от описательной статистики до проверки гипотез.
