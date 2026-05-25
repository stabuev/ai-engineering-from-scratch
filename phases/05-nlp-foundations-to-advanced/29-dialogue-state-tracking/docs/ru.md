# Отслеживание состояния диалога

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian." Три реплики, три обновления состояния. DST поддерживает slot-value dict в актуальном виде, чтобы бронирование сработало.

**Тип:** Практика
**Языки:** Python
**Предварительные требования:** Фаза 5 · 17 (Chatbots), Фаза 5 · 20 (Structured Outputs)
**Время:** ~75 минут

## Проблема

В task-oriented dialogue system цель пользователя кодируется как набор пар slot-value: `{cuisine: italian, area: north, price: moderate}`. Каждая реплика пользователя может добавить, изменить или удалить слот. Система должна прочитать весь разговор и корректно вывести текущее состояние.

Ошибитесь в одном слоте, и система забронирует не тот ресторан, запланирует не тот рейс или спишет деньги не с той карты. DST — шарнир между тем, что сказал пользователь, и тем, что выполняет backend.

Почему это все еще важно в 2026 году, несмотря на LLM:

- Домены, чувствительные к compliance (банкинг, здравоохранение, бронирование авиабилетов), требуют детерминированных значений слотов, а не free-form генерации.
- Агентам с использованием инструментов все еще нужно разрешить слоты перед вызовом API.
- Многоходовая коррекция сложнее, чем кажется: "actually no, make it Thursday."

Современный пайплайн: классические концепции DST + LLM-экстракторы + guardrails структурированного вывода.

## Концепция

![DST: история диалога → состояние slot-value](../assets/dst.svg)

**Структура задачи.** Схема задает домены (restaurant, hotel, taxi) и их слоты (cuisine, area, price, people). Каждый слот может быть пустым, заполненным значением из закрытого множества (price: {cheap, moderate, expensive}) или free-form значением (name: "The Copper Kettle").

**Две формулировки DST.**

- **Классификация.** Для каждой пары (slot, candidate_value) предсказать yes/no. Работает для слотов с закрытым словарем. Стандарт до 2020 года.
- **Генерация.** По диалогу сгенерировать значения слотов как свободный текст. Работает для слотов с открытым словарем. Современный вариант по умолчанию.

**Метрика.** Joint Goal Accuracy (JGA) — доля реплик, где *каждый* слот корректен. Все или ничего. Лидеры MultiWOZ 2.4 в 2026 году достигают примерно 83%.

**Архитектуры.**

1. **Rule-based (slot regex + keyword).** Сильный baseline для узких доменов. Удобно отлаживать.
2. **TripPy / BERT-DST.** Генерация на основе копирования с BERT-кодированием. Стандарт до LLM.
3. **LDST (LLaMA + LoRA).** Instruction-tuned LLM с prompting по domain-slot. Достигает качества уровня ChatGPT на MultiWOZ 2.4.
4. **Ontology-free (2024–26).** Пропустить схему; генерировать имена и значения слотов напрямую. Поддерживает открытые домены.
5. **Prompt + structured output (2024–26).** LLM со схемой Pydantic + constrained decoding. 5 строк кода, готово для продакшена.

### Классические режимы отказа

- **Кореференция между репликами.** "Let's stay with the first option." Нужно понять, какой вариант имеется в виду.
- **Перезаписать или добавить.** Пользователь говорит "add Italian." Нужно заменить cuisine или добавить?
- **Неявные подтверждения.** "OK cool" — это приняло предложенное бронирование?
- **Коррекция.** "Actually make it 7 pm." Нужно обновить time, не очищая остальные слоты.
- **Кореференция к предыдущей реплике системы.** "Yes, that one." Какой именно "that"?

## Соберите это

### Шаг 1: rule-based извлекатель слотов

См. `code/main.py`. Regex + словари синонимов покрывают 70% канонических высказываний в узких доменах:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Хрупко за пределами канонического словаря. Работает для детерминированных подтверждений слотов.

### Шаг 2: цикл обновления состояния

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Три инварианта:

- Никогда не сбрасывайте слот, которого пользователь не касался.
- Явное отрицание ("never mind the cuisine") должно очищать слот.
- Коррекция пользователя ("actually...") должна перезаписывать, а не добавлять.

### Шаг 3: DST на основе LLM со структурированным выводом

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic гарантируют валидный объект состояния. Без regex, без несовпадений схемы, без галлюцинированных слотов.

### Шаг 4: оценка JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Калибруйте: на какой доле реплик система получает ВСЕ слоты правильно? Для MultiWOZ 2.4 лучшие системы 2026 года: 80-83%. Ваша in-domain система должна превосходить это на вашем узком словаре, иначе LLM-baseline вас обгоняет.

### Шаг 5: обработка коррекции

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

При обнаруженной коррекции перезаписывайте последний обновленный слот, а не добавляйте. Трудно сделать правильно без помощи LLM. Современный паттерн: всегда позволяйте LLM регенерировать все состояние из истории, а не обновлять инкрементально — это естественно обрабатывает коррекции.

## Подводные камни

- **Стоимость регенерации по полной истории.** Если LLM регенерирует состояние на каждой реплике, это стоит O(n²) токенов суммарно. Ограничивайте историю или суммаризируйте старые реплики.
- **Дрейф схемы.** Добавление новых слотов post-hoc ломает старые обучающие данные. Версионируйте схему.
- **Чувствительность к регистру.** "Italian" vs "italian" vs "ITALIAN" — нормализуйте везде.
- **Неявное наследование.** Если пользователь ранее указал "for 4 people," новый запрос на другое время не должен очищать people. Всегда передавайте полную историю.
- **Free-form vs closed-set.** Имена, времена и адреса требуют free-form слотов; кухни и районы закрыты. Смешивайте оба типа в схеме.

## Используйте это

Стек 2026 года:

| Ситуация | Подход |
|-----------|----------|
| Узкий домен (один или два intents) | Rule-based + regex |
| Широкий домен, доступны размеченные данные | LDST (LLaMA + LoRA на данных в стиле MultiWOZ) |
| Широкий домен, нет меток, готово к prod | LLM + Instructor + Pydantic schema |
| Речь / голос | ASR + normalizer + LLM-DST |
| Multi-domain booking flow | Schema-guided LLM с Pydantic-моделями по доменам |
| Compliance-sensitive | Rule-based primary, LLM fallback с confirmation flow |

## Отгрузите это

Сохраните как `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Упражнения

1. **Легко.** Постройте rule-based state tracker в `code/main.py` для 3 слотов (cuisine, area, price). Проверьте на 10 hand-crafted диалогах. Измерьте JGA.
2. **Средне.** Тот же датасет с Instructor + Pydantic + небольшой LLM. Сравните JGA. Изучите самые сложные реплики.
3. **Сложно.** Реализуйте оба варианта и маршрутизируйте: rule-based primary, LLM fallback, когда rule-based выдает <2 слотов с confidence. Измерьте combined JGA и inference cost per turn.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| DST | Dialogue state tracking | Поддерживать slot-value dict между репликами диалога. |
| Slot | Единица намерения пользователя | Именованный параметр, который нужен backend (cuisine, date). |
| Domain | Область задачи | Restaurant, hotel, taxi — наборы слотов. |
| JGA | Joint Goal Accuracy | Доля реплик, где каждый слот корректен. Все или ничего. |
| MultiWOZ | Бенчмарк | Multi-domain WOZ dataset; стандартная оценка DST. |
| Ontology-free DST | Нет схемы | Генерировать имена и значения слотов напрямую, без фиксированного списка. |
| Correction | "Actually..." | Реплика, которая перезаписывает ранее заполненный слот. |

## Дополнительное чтение

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — канонический бенчмарк.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — instruction tuning LLaMA + LoRA для DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — рабочая лошадка DST на основе копирования.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — unsupervised TOD на основе EM.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — канонические результаты DST.
