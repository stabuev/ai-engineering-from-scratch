# Чатботы — от правил к нейронным моделям и LLM-агентам

> ELIZA отвечала совпадениями шаблонов. DialogFlow сопоставлял intents. GPT отвечал из весов. Claude запускает инструменты и проверяет. Каждая эпоха решала худший провал предыдущей.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 5 · 13 (Question Answering), Фаза 5 · 14 (Information Retrieval)
**Время:** ~75 минут

## Проблема

Пользователь говорит: "I want to change my flight." Система должна понять, чего он хочет, какой информации не хватает, как ее получить и как выполнить действие. Затем пользователь говорит: "wait, what if I cancel instead?", и система должна помнить контекст, переключить задачу и сохранить состояние.

Диалог сложен для ML-системы. Вход открыт и не ограничен. Выход должен оставаться связным на протяжении многих ходов. Системе может понадобиться действовать в мире (изменить рейс, списать деньги с карты). Каждый неверный шаг виден пользователю.

Архитектуры чатботов прошли через четыре парадигмы, каждая из которых появилась потому, что предыдущая слишком заметно проваливалась. Этот урок проходит по ним по порядку. Production-ландшафт 2026 года — гибрид двух последних.

## Концепция

![Эволюция чатботов: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**На основе правил (ELIZA, AIML, DialogFlow).** Написанные вручную шаблоны сопоставляются с пользовательским вводом и порождают ответы. Классификаторы intents направляют в предопределенные flows. Конечные автоматы slot filling собирают нужную информацию. Великолепно работает внутри узкой области, для которой было спроектировано. Немедленно ломается за ее пределами. Все еще поставляется в safety-critical доменах (банковская аутентификация, бронирование авиабилетов), где галлюцинации недопустимы.

**Retrieval-based.** Система в стиле FAQ. Кодирует каждую пару (utterance, response). Во время выполнения кодирует сообщение пользователя и извлекает ближайший сохраненный ответ. Представьте классическую функцию Zendesk "similar articles". Лучше правил обрабатывает перефразирования. Генерации нет, поэтому нет галлюцинаций.

**Нейронные (seq2seq).** Encoder-decoder, обученный на логах диалогов. Генерирует ответы с нуля. Беглый, но склонен к общим ответам ("I don't know") и фактическому дрейфу. Никогда не оставался надежно по теме. Причина, по которой у Google, Facebook и Microsoft были разочаровывающие чатботы в 2016-2019 годах.

**LLM-агенты.** Языковая модель, обернутая в цикл, который планирует, вызывает инструменты и проверяет результаты. Это не чатбот с длинным prompt. Агентный цикл: plan → call tool → observe result → decide next step. Retrieval-first grounding (RAG) удерживает его от галлюцинаций. Tool calls позволяют ему реально что-то делать. Это архитектура 2026 года.

Четыре парадигмы не являются последовательными заменами. Production-чатбот 2026 года маршрутизирует через все четыре: правила для аутентификации и разрушительных действий, retrieval для FAQ, нейронная генерация для естественных формулировок, LLM-агент для неоднозначных открытых запросов.

## Соберите это

### Шаг 1: сопоставление шаблонов на правилах

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA в 20 строк. Трюк с отражением ("I feel sad" → "Why do you feel sad") — каноническая демонстрация психотерапевта из Weizenbaum 1966. Все еще поучительно.

### Шаг 2: retrieval-based (FAQ)

Этот иллюстративный фрагмент требует `pip install sentence-transformers` (что подтягивает torch). Запускаемый `code/main.py` для этого урока вместо этого использует Jaccard similarity из stdlib, поэтому урок запускается без внешних зависимостей.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Отказ по порогу — ключевое проектное решение. Если лучшее совпадение недостаточно близко, верните `None` и позвольте системе эскалировать.

### Шаг 3: нейронная генерация (бейзлайн)

Используйте небольшой instruction-tuned encoder-decoder (FLAN-T5) или дообученную разговорную модель. В 2026 году сама по себе непригодна для production (противоречия, уход с темы, фактическая бессмыслица), но поставляется внутри гибридных систем для естественных формулировок. Decoder-only модели в стиле DialoGPT требуют явных разделителей ходов и обработки EOS, чтобы давать связные ответы; pipeline FLAN-T5 text2text работает из коробки для учебного примера.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Шаг 4: цикл LLM-агента

Production-форма 2026 года:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            result = tools[tool_name](**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Нужно назвать три вещи. Tools — вызываемые функции, которые LLM может запускать. Цикл завершается, когда LLM возвращает финальный ответ вместо tool call. Бюджет шагов предотвращает бесконечные циклы на неоднозначных задачах.

Реальный production добавляет: retrieval-first grounding (подмешивать релевантные документы перед каждым вызовом LLM), guardrails (отказывать в разрушительных действиях без подтверждения), observability (логировать каждый шаг) и evaluations (автоматические проверки, что поведение агента остается в рамках спецификации).

### Шаг 5: гибридная маршрутизация

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Паттерн: детерминированные правила для всего разрушительного, retrieval для заготовленных FAQ, LLM-агенты для всего остального. Именно это поставляется в системах customer support в 2026 году.

## Используйте это

Стек 2026 года:

| Сценарий | Архитектура |
|---------|---------------|
| Бронирование, оплата, аутентификация | Rule-based state machines + slot filling |
| FAQ поддержки клиентов | Retrieval по курируемым ответам |
| Открытый чат помощи | LLM agent with RAG + tool calls |
| Внутренние инструменты / IDE assistants | LLM agent with tool calls (search, read, write) |
| Companion / character chatbots | Настроенная LLM с persona system prompt, retrieval по знаниям |

Всегда используйте гибридную маршрутизацию в production. Ни одна архитектура не обрабатывает хорошо каждый запрос. Сам слой маршрутизации обычно представляет собой небольшой классификатор intent.

## Failure modes, которые все еще попадают в поставку

- **Уверенная фабрикация.** LLM-агент утверждает, что выполнил действие, которого не выполнял. Митигация: проверять результаты, логировать tool calls, никогда не позволять LLM утверждать, что что-то сделано, без успешного возврата инструмента.
- **Prompt injection.** Пользователь вставляет текст, который переопределяет system prompt. Занимает место LLM01 в OWASP Top 10 for LLM Applications 2025. Два вида: direct injection (вставлена в чат) и indirect injection (скрыта в документах, письмах или outputs инструментов, которые читает агент).

  Частоты атак зависят от сценария. Измеренные success rates находятся в диапазоне ~0.5-8.5% среди frontier models в общих бенчмарках tool-use и coding. Конкретные высокорисковые настройки (адаптивные атаки против AI coding agents, уязвимая orchestration) достигали ~84%. Production CVEs включают EchoLeak (CVE-2025-32711, CVSS 9.3) — zero-click data-exfiltration уязвимость в Microsoft 365 Copilot, запускаемая email под контролем атакующего.

  Митигации: рассматривать пользовательский ввод как недоверенный на всем протяжении цикла; санитизировать перед tool calls; изолировать outputs инструментов от основного prompt; использовать паттерн Plan-Verify-Execute (PVE), где агент сначала планирует, затем проверяет каждое действие относительно этого плана перед выполнением (это останавливает внедрение новых незапланированных действий через результаты инструментов); требовать подтверждение пользователя для разрушительных действий; применять least privilege к областям доступа инструментов.

  Никакое prompt engineering полностью не устраняет этот риск. Требуются внешние runtime defense layers (LLM Guard, allowlist validation, semantic anomaly detection).
- **Scope creep.** Агент уходит с задачи, потому что tool call вернул информацию, связанную лишь косвенно. Митигация: сужать контракты инструментов; держать system prompt сфокусированным; добавить evaluations для off-task rate.
- **Бесконечные циклы.** Агент продолжает вызывать один и тот же инструмент. Митигация: бюджет шагов, дедупликация tool calls, LLM judge на вопрос "are we making progress."
- **Исчерпание context window.** Длинные диалоги выталкивают самые ранние ходы из контекста. Митигация: суммаризировать старые ходы, извлекать релевантные прошлые ходы по similarity или использовать long-context model.

## Доведите до поставки

Сохраните как `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Упражнения

1. **Легко.** Реализуйте приведенный выше rule-based respond с 10 шаблонами для бота заказа в кофейне. Проверьте крайние случаи: двойные заказы, изменения, отмена, неясный intent.
2. **Средне.** Постройте гибрид FAQ + LLM fallback. 50 заготовленных FAQ-записей для SaaS-продукта, LLM fallback с retrieval по сайту документации. Измерьте refusal rate и accuracy на 100 реальных вопросах поддержки.
3. **Сложно.** Реализуйте приведенный выше agent loop с тремя инструментами (search, read-user-data, send-email). Запустите evaluation с 50 тестовыми сценариями, включая попытки prompt injection. Сообщите off-task rate, failed task rate и любой успех injection.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| Intent | Чего хочет пользователь | Категориальная метка (book_flight, reset_password). Маршрутизируется в handler. |
| Slot | Часть информации | Параметр, который нужен боту (date, destination). Slot filling — последовательность вопросов. |
| RAG | Retrieval плюс generation | Извлечь релевантные документы, затем заземлить ответ LLM. |
| Tool call | Вызов функции | LLM испускает структурированный вызов с name + args. Runtime выполняет его и возвращает результат. |
| Agent loop | Планировать, действовать, проверять | Контроллер, который запускает вызовы LLM вперемежку с tool calls до завершения задачи. |
| Prompt injection | Пользователь атакует prompt | Вредоносный ввод, который пытается переопределить system prompt. |

## Дополнительное чтение

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — оригинальная статья о rule-based chatbot.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — поздняя статья Google о neural chatbot, прямо перед тем как LLM-агенты заняли поле.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — статья, которая назвала паттерн agent loop.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — production-рекомендации 2024 года, которые все еще справедливы в 2026.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — статья о prompt injection.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — рейтинг, сделавший prompt injection главной проблемой безопасности.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — практические защиты на orchestration layer, включая Plan-Verify-Execute и user-confirmation flows.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — каноническая zero-click data-exfiltration CVE из indirect prompt injection. Референсный случай, показывающий, почему write-access agents требуют runtime defenses.
