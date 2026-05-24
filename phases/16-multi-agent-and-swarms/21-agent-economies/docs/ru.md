# Агентные экономики, токеновые стимулы, репутация

> Автономным агентам с длинным горизонтом работы (кривая METR от 1 до 8 часов) нужна экономическая агентность. Формирующийся **5-слойный стек**: **DePIN** (физические вычисления) → **Identity** (W3C DIDs + репутационный капитал) → **Cognition** (RAG + MCP) → **Settlement** (account abstraction) → **Governance** (Agentic DAOs). Производственные сети агентных стимулов включают **Bittensor** (подсети TAO вознаграждают модели для конкретных задач), **Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + токен FET) и **Gonka** (transformer-based PoW, перераспределяющий вычисления на продуктивные AI-задачи). Академическая работа: децентрализованная LaMAS из AAMAS 2025 использует **Shapley-value credit attribution** для справедливого вознаграждения агентов-участников; Google Research "Mechanism design for large language models" предлагает **token auctions** с оплатой по второй цене при монотонной агрегации. В этом уроке мы строим минимальный агентный marketplace, применяем атрибуцию вклада по значению Шепли к multi-agent pipeline и запускаем token auction второй цены, чтобы аппарат теории игр стал конкретным.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Время:** ~75 минут

## Проблема

Multi-agent системы усложняются, когда агенты создают ценность совместно, но вознаграждать их нужно индивидуально. Классические механизмы — равное деление, "последний участник получает все" — несправедливы или поддаются манипуляциям. Вознаграждение на основе коалиций через значения Шепли справедливо по построению, но дорого вычисляется. Литература 2025-2026 продвигает полезные приближения: Shapley sampling, аукционы с монотонной агрегацией и on-chain репутацию, которая накапливается из подтвержденных вкладов.

Помимо атрибуции вклада, область перешла к реальным экономическим агентам: Bittensor TAO вознаграждает mining compute для дообучения моделей конкретных подсетей, Fetch.ai/ASI вознаграждает использование ASI-1 Mini LLM токенами FET, Gonka перераспределяет transformer proof-of-work в сторону продуктивных AI-задач. Агенты, которые автономно совершают транзакции, существуют уже сегодня; вопрос в том, как выровнять стимулы.

Этот урок рассматривает агентные экономики как конкретное семейство задач — атрибуция вклада, mechanism design и репутация — и строит каждую часть с минимальной математикой, чтобы идеи закрепились.

## Концепция

### 5-слойный стек агентной экономики

1. **DePIN (physical compute).** Децентрализованная инфраструктура, сдающая в аренду GPU, хранилище, пропускную способность. Bittensor subnets, Render Network, Akash. Не специфична для агентов; агенты ее используют.
2. **Identity.** W3C Decentralized Identifiers (DIDs) дают каждому агенту устойчивый ID, независимый от платформы. Репутация накапливается у DID. Agent Network Protocol (ANP) использует DID как слой обнаружения.
3. **Cognition.** Цикл рассуждения агента: LLM + RAG + MCP. Именно это строят остальные фазы.
4. **Settlement.** Account abstraction (ERC-4337) позволяет агентам платить gas со своих собственных балансов без хранения ETH. Агенты могут платить за сервисы, друг другу или за вычисления.
5. **Governance.** Agentic DAOs: структуры управления, где люди *и* агенты голосуют за изменения протокола, а voting power привязана к репутации.

Не каждая production-система использует все пять слоев. Bittensor использует 1, 2, частично 3, частично 4, и не использует 5. OpenAI agents не используют ничего, кроме 3. Стек — это справочная карта, а не требование.

### Bittensor, Fetch.ai, Gonka — что работает

**Bittensor (TAO).** Подсети — это специализированные задачи (language modeling, image generation, forecasting). Miners отправляют outputs моделей. Validators ранжируют их; stake-weighted scoring распределяет вознаграждения TAO. У каждой подсети своя evaluation. Экономический урок: платите за качество output для конкретной задачи, а не за использованные вычисления.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM работает в сети Fetch.ai; пользователи платят токенами FET за inference. Нарратив agents-as-peers здесь сильнее: агент в Fetch может вызвать другого агента для задачи и заплатить в FET.

**Gonka.** Transformer proof-of-work: "работа" — это forward passes трансформера. Miners зарабатывают, выполняя inference tasks с известными правильными outputs (из training data). Ресурсно-продуктивный PoW вместо hash-based PoW.

Все три являются production-grade по состоянию на апрель 2026. Распределение payoff различается. Bittensor вознаграждает качество относительно subnet validators; Fetch вознаграждает полезность, измеряемую платящими пользователями; Gonka вознаграждает проверяемую inference work.

### Shapley-value credit attribution

Три агента совместно работают над задачей. Вывод получает score 0.8. Кто сколько внес?

Значение Шепли: единственное распределение вклада, удовлетворяющее четырем аксиомам (efficiency, symmetry, linearity, null). Для агента `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

где `S_i_O` — множество агентов перед `i` в ordering `O`. На практике: перечислить все перестановки, записать marginal contribution каждого агента в каждой перестановке, усреднить.

Для N=3 агентов есть 6 перестановок. Для N=10 — 3.6M, поэтому на практике вы сэмплируете orderings, а не перечисляете их.

### Аукцион второй цены для агрегации

Google Research ("Mechanism design for large language models") предлагает token auctions второй цены для агрегации LLM outputs. Setup: N агентов предлагают completion; у каждого есть private value за то, чтобы быть выбранным. Auctioneer выбирает предложение с наибольшей value и платит *вторую по величине* value. При монотонной агрегации (value зависит от того, какое proposal выбрано, а не от того, сколько bids было сделано) это truthfulness — агентам выгодно bid their true value.

Почему это важно для LLM-систем: вы можете отдавать completion tasks нескольким агентам с разными ценами; аукцион выбирает лучшее + платит справедливо, и у агентов нет стимула искажать информацию.

### Репутационный капитал

DID-bound репутационный score накапливается из подтвержденных вкладов. Простое правило обновления:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

С decay factor `alpha`, близким к 1. Репутация:

- Дешева для чтения в routing decisions ("отправляй сложные задачи агентам с высокой rep").
- Дорога для подделки (накапливается со временем, привязана к DID).
- Может быть slashed: вклады, не прошедшие verification, вычитаются.

### AAMAS 2025 decentralized LaMAS

Предложение LaMAS (AAMAS 2025) объединяет: DID identity, Shapley-value credit attribution и простой auction mechanism. Ключевое утверждение: децентрализация шага credit attribution делает систему аудируемой и устойчивой к single-point manipulation.

### Где экономика разваливается

- **Price oracle manipulation.** Если credit function можно заиграть, агенты будут ее заигрывать. Каждому механизму нужен adversarial test.
- **Sybil attacks.** Один оператор запускает N фальшивых агентов, чтобы раздуть собственный вклад. DIDs замедляют, но не останавливают это; mitigation — cost-to-forge репутации.
- **Verification cost.** Атрибуция вклада справедлива ровно настолько, насколько справедлив verifier. Если verification дешевый (маленькая LLM), его можно обмануть; если дорогой (human panel), система не масштабируется.
- **Regulatory overhang.** Агентные экономики пересекаются с финансовым регулированием. Bittensor, Fetch и Gonka по состоянию на 2026 работают в правовых серых зонах в некоторых юрисдикциях.

### Когда агентные экономики имеют смысл

- **Open networks with heterogeneous operators.** Ни одна команда не контролирует всех агентов.
- **Verifiable outputs.** Без verification атрибуция вклада — догадка.
- **Long-horizon workflows.** Одноразовые задачи не выигрывают от накопления репутации.
- **Tokenized payments are legally viable** в вашей юрисдикции.

В закрытых корпоративных системах экономика уступает место более простому распределению (менеджеры назначают работу, метрики внутренние). Экономическая литература в основном применима к открытым сетям.

## Соберите

`code/main.py` реализует:

- `shapley(value_fn, agents)` — точное вычисление Shapley через перечисление для малых N.
- `second_price_auction(bids)` — truthful mechanism; победитель платит second-highest.
- `Reputation` — DID-bound репутация с exponential decay и slashing.
- Demo 1: три агента сотрудничают, exact Shapley attributes credit.
- Demo 2: пять агентов bid за task slot; аукцион второй цены выбирает winner + payment.
- Demo 3: 100 раундов назначения задач агентам с неоднородной rep; rep-weighted routing превосходит random.

Запуск:

```
python3 code/main.py
```

Ожидаемый output: Shapley values для каждого агента; auction result, показывающий truthful-bid equilibrium; rep-weighted routing, показывающий 10-20% прирост качества над random после warmup.

## Используйте

`outputs/skill-economy-designer.md` проектирует минимальную агентную экономику: выбор identity layer, credit attribution mechanism, payment mechanism, reputation rule.

## Запустите в production

Запуск агентной экономики в 2026:

- **Начинайте с репутации, а не с токенов.** Репутацию дешево реализовать, и она ценна сама по себе; токены добавляют правовую и экономическую сложность.
- **Verify before you reward.** Никогда не распределяйте credit без независимого verification step. Self-reported quality накапливает sybil games.
- **Shapley-sample, not Shapley-exact.** Сэмплируйте 100-1000 orderings; exact enumeration не масштабируется.
- **Ограничивайте decay factor и задавайте floor reputation.** Безграничный decay стирает легитимных contributors; слишком медленный decay вознаграждает stale high-rep agents.
- **Аудируйте механизмы adversarially.** Запускайте red-team scenarios перед открытием сети. У каждого механизма есть game theory; вам нужно найти дыры, а не attackers.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что Shapley values суммируются в total value (efficiency axiom). Измените value function; меняются ли Shapley allocations в ожидаемом направлении?
2. Реализуйте Shapley *sampling* (Monte Carlo over K orderings). Как K влияет на approximation accuracy? Сравните с exact для N=4.
3. Реализуйте coalition-forming step перед auction: агенты могут объединяться в teams и bid как единое целое. Какие coalitions формируются? Является ли outcome Pareto-better, чем individual bidding?
4. Прочитайте пост Google Research по mechanism-design. Найдите одно предположение, нарушение которого ломает truthfulness. Как выглядит такой failure mode в LLM setting?
5. Прочитайте paper AAMAS 2025 decentralized LaMAS. Реализуйте их Shapley step для 10 агентов на synthetic task. Сколько времени занимает exact computation? Насколько близко sampling с 100 draws?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| DePIN | "Decentralized physical infrastructure" | Token-incentivized compute/storage/bandwidth. Bittensor, Akash, Render. |
| DID | "Decentralized identifier" | W3C spec для portable IDs. Репутация агента привязана к DID, а не к платформе. |
| ERC-4337 | "Account abstraction" | Contract accounts, которые могут sponsor gas, включая agent payments. |
| Shapley value | "Fair credit attribution" | Unique allocation, удовлетворяющее efficiency, symmetry, linearity, null. |
| Second-price auction | "Vickrey auction" | Truthful mechanism: winner pays second-highest bid. Совместим с monotone aggregation. |
| Reputation capital | "Accumulated quality score" | DID-bound score из confirmed contributions; decays over time. |
| Agentic DAO | "Agents + humans govern" | DAO с agent voters как first-class участниками, voting power tied to reputation. |
| TAO / FET / GPU credits | "Token denominations" | Bittensor TAO, Fetch.ai FET, различные DePIN tokens. |

## Дополнительное чтение

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — обзор 2026 года 5-слойного стека агентной экономики
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — token auctions with monotone aggregation
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value credit attribution
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet structure and reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM and FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — identity foundation
