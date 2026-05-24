# Compliance — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Multi-framework coverage — базовое требование для enterprise deals в 2026. **EU AI Act**: in force since August 1, 2024. Большинство high-risk requirements enforce August 2, 2026. Штрафы до €15M или 3% global annual turnover за high-risk-system obligations (Art. 99(4)); до €35M или 7% за prohibited AI practices (Art. 99(3)). Применяется глобально, если вы обслуживаете EU users. **Colorado AI Act**: effective June 30, 2026 (delayed from February 2026 by SB25B-004) — impact assessments для high-risk systems, right to appeal AI decisions. Virginia similar for credit/employment/housing/education. **SOC 2 Type II**: de facto B2B AI requirement (Type II, not Type I, for fintech). **GDPR**: largest documented AI-specific fine is €30.5M against Clearview AI (Dutch DPA, Sept 2024); Italy's Garante issued €15M against OpenAI in Dec 2024 (later overturned on appeal in March 2026). Real-time PII redaction at inference — defensible standard; post-processing cleanup недостаточно. **HIPAA**: healthcare bound — нельзя отправлять PHI во external AI services без BAA. **PCI-DSS**: AI-interaction-layer coverage требует configuration + contractual agreements, не automatic. **ISO 42001**: emerging AI governance standard, растущее procurement requirement наряду с ISO 27001. Reference profile: OpenAI maintains SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS for ChatGPT payment components. Cross-framework mapping снижает audit fatigue: access controls map across ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a).

**Тип:** Learn
**Языки:** (Python optional — compliance is policy + process, not code)
**Предварительные требования:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Время:** ~60 minutes

## Цели обучения

- Перечислить семь frameworks 2026, релевантных LLM products, и сопоставить каждый с customer segment.
- Процитировать EU AI Act enforcement timeline (in force August 2024; high-risk enforcement August 2026) и two-tier fine ceiling (€15M / 3% for high-risk obligations, €35M / 7% for prohibited practices).
- Объяснить, почему post-processing PII cleanup недостаточен для GDPR, и назвать real-time inference-layer redaction как defensible standard.
- Описать cross-framework control mapping (e.g., access control maps to ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)).

## Проблема

Enterprise customer procurement просит SOC 2 Type II, GDPR, HIPAA BAA, ISO 27001 и "EU AI Act compliance statement." У вашей команды есть SOC 2 Type I. До Type II — шесть месяцев, а GDPR Article 30 records вы еще не начинали.

Multi-framework coverage — не проблема LLM, а проблема enterprise-SaaS с LLM-specific overlays. Procurement teams в 2026 хотят matrix с row per framework и column per control, а не PDF.

## Концепция

### Семь frameworks

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls audited over 6-12 months |
| HIPAA | US healthcare | BAA required; PHI cannot leave infrastructure without signed agreement |
| GDPR | EU users | Real-time PII redaction; data subject rights; Article 30 records |
| PCI-DSS | Payment data | Configuration + contracts for AI touching payment |
| EU AI Act | Serving EU users | Risk tier classification; high-risk systems: conformity assessment, documentation, logging |
| Colorado AI Act | Serving CO residents | Impact assessments; right to appeal |
| ISO 42001 | AI governance | Emerging; pairs with ISO 27001 |

### EU AI Act timeline

- August 1, 2024: in force.
- February 2, 2025: prohibited-AI practices enforced.
- August 2, 2026: high-risk systems enforced (conformity assessment, documentation, logging).
- August 2027: high-risk systems in products under harmonized legislation.

Risk tiers: Unacceptable (banned), High-risk (conformity + logging), Limited-risk (transparency), Minimal-risk (no constraint). Большинство B2B LLM SaaS — limited-risk; high-risk включается для employment, credit, education, law enforcement, migration, essential services.

Fines (Article 99): до €15M или 3% global annual turnover за breaches of high-risk-system obligations (Art. 99(4)); до €35M или 7% за prohibited AI practices (Art. 99(3)); whichever higher applies.

### GDPR — real-time redaction is the standard

Post-processing cleanup (редактировать PII после того, как LLM ее увидела) — не defensible posture: model уже видела данные. Real-time inference-layer redaction — standard 2026:

- Entity recognition before the LLM call.
- Consistent tokenization (Mesh approach) preserves semantics.
- Store only redacted prompts + consented opt-in raw.

Recent enforcement: €30.5M against Clearview AI (Dutch DPA, Sept 2024) — largest documented AI-specific GDPR fine to date; €15M against OpenAI (Italy's Garante, Dec 2024) — largest LLM-specific fine, though it was overturned on appeal in March 2026 and the ruling remains under further review. Post-processing claims have failed at audit.

### HIPAA — BAA is not optional

Нельзя отправлять PHI во external AI services без signed Business Associate Agreement. All three hyperscaler LLM platforms (Bedrock, Azure OpenAI, Vertex) offer BAAs. OpenAI direct API offers BAA. Anthropic direct API offers BAA. Confirm before sending PHI.

### SOC 2 Type II

Type I: controls designed and documented.
Type II: controls operate effectively over 6-12 months.

B2B procurement в 2026 по умолчанию требует Type II. Type I — starter; Type II — gate.

Common audit drivers: access logs (who saw what), change management (how was it deployed), risk assessments (quarterly), incident response (tested?). Audit log из Phase 17 · 25 directly reusable.

### Cross-framework mapping

Одна access control policy удовлетворяет controls нескольких frameworks:

| Control | Frameworks |
|---------|-----------|
| Access logging | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| Encryption in transit | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Compliance tools (Drata, Vanta, Secureframe) автоматизируют это mapping. На масштабе стоят своих денег.

### ISO 42001 — emerging

Published late 2023. Растущее procurement requirement alongside ISO 27001. Framework for AI governance including risk management, data quality, transparency, human oversight.

### OpenAI's reference profile

OpenAI maintains SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS for ChatGPT payment components. Это примерно enterprise table stakes в 2026.

### Числа, которые стоит запомнить

- EU AI Act fines: up to €15M / 3% (high-risk obligations, Art. 99(4)); up to €35M / 7% (prohibited practices, Art. 99(3)).
- EU AI Act high-risk enforcement: August 2, 2026.
- Largest documented AI-specific GDPR fine: €30.5M, Clearview AI (Dutch DPA, Sept 2024).
- Largest LLM-specific GDPR fine: €15M, OpenAI (Italy's Garante, Dec 2024; overturned on appeal March 2026).
- SOC 2 Type II window: 6-12 months of operated controls.
- Colorado AI Act effective date: June 30, 2026 (delayed from February 2026 by SB25B-004).

## Используйте это

`code/main.py` — compliance-mapping spreadsheet in Python: по control перечисляет frameworks, которые он удовлетворяет.

## Отгрузите это

Этот урок создает `outputs/skill-compliance-matrix.md`. По customer segment и geography задает required frameworks and controls.

## Упражнения

1. Ваш первый enterprise customer требует SOC 2 Type II, HIPAA BAA, EU AI Act statement. Какова minimum viable compliance posture, чтобы выиграть deal?
2. Классифицируйте три hypothetical LLM products under EU AI Act risk tiers. Что меняется at high-risk?
3. Вы случайно отправили PHI provider без BAA. Пройдите incident response.
4. Аргументируйте, является ли ISO 42001 "necessary in 2026" для mid-market AI vendor.
5. Сопоставьте поля вашего LLM audit log (Phase 17 · 25) минимум с тремя framework controls.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| SOC 2 Type II | "audited controls" | Controls operating over 6-12 months, independently attested |
| HIPAA BAA | "healthcare contract" | Business Associate Agreement; required for PHI |
| GDPR | "EU privacy" | Real-time PII redaction is the defensible 2026 standard |
| EU AI Act | "EU AI rules" | High-risk enforcement August 2026; €15M / 3% (high-risk obligations) — €35M / 7% (prohibited practices) |
| Colorado AI Act | "US AI state law" | June 30, 2026 effective (delayed by SB25B-004); impact assessments |
| ISO 42001 | "AI governance" | Emerging framework for AI risk + transparency |
| ISO 27001 | "security ISMS" | Information Security Management System baseline |
| Conformity assessment | "EU AI doc package" | High-risk requirement: docs, testing, logging |
| Cross-framework mapping | "one control, many frames" | Single policy satisfies multiple framework controls |

## Дополнительное чтение

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.
