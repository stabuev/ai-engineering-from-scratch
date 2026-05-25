# Водяные знаки — SynthID, Stable Signature, C2PA

> Три технологии структурируют provenance AI-generated content в 2026 году. SynthID (Google DeepMind) — водяные знаки для изображений, запущены в августе 2023; текст+видео в мае 2024 (Gemini + Veo); текстовая часть open-sourced в октябре 2024 через Responsible GenAI Toolkit; единый multi-media detector в ноябре 2025 вместе с Gemini 3 Pro. Text watermarking незаметно корректирует вероятности сэмплирования следующего токена; водяные знаки изображений/видео переживают сжатие, обрезку, фильтры, изменения frame-rate. Stable Signature (Fernandez et al., ICCV 2023, arXiv:2303.15435) — дообучает decoder латентной диффузии так, что каждый output содержит фиксированное сообщение; обрезанные (10% content) сгенерированные изображения обнаруживаются >90% при FPR<1e-6. Последующая работа "Stable Signature is Unstable" (arXiv:2405.07145, май 2024) — fine-tuning удаляет водяной знак при сохранении качества. C2PA — криптографически подписанный, tamper-evident стандарт metadata (C2PA 2.2 Explainer 2025). Watermarking и C2PA дополняют друг друга: metadata можно удалить, но она несет более богатый provenance; водяные знаки сохраняются при transcoding, но несут меньше информации.

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## Цели обучения

- Описать token-level watermarking (в стиле SynthID-text) и механизм, благодаря которому он обнаружим.
- Описать Stable Signature и атаку удаления 2024 года, которая его сломала.
- Сформулировать роль C2PA и почему он дополняет watermarking.
- Описать ключевые ограничения: model-specific signal, устойчивость к paraphrase и meaning-preserving attacks (arXiv:2508.20228).

## Проблема

В 2023-2024 deepfakes и AI-generated content массово вошли в политические и потребительские контексты. Watermarking — предлагаемый технический сигнал provenance: помечать генерации в момент создания, обнаруживать позже. Свидетельства 2025 года: ни один водяной знак не является безусловно устойчивым, но в слое с C2PA metadata комбинация дает пригодную историю provenance.

## Концепция

### Text watermarking (в стиле SynthID-text)

Механизм Kirchenbauer et al. 2023, доведенный Google до production:

1. На каждом шаге декодирования хэшировать предыдущие K токенов, чтобы получить псевдослучайное разбиение словаря на "green" и "red" множества.
2. Смещать сэмплирование к green set, добавляя δ к green logits.
3. Генерация содержит больше green tokens, чем получилось бы случайно.

Обнаружение: заново хэшировать каждый prefix, считать green tokens в генерации, вычислять z-score. z-score >0 для текста с водяным знаком, ~0 для человеческого текста.

Свойства:
- Незаметен для читателей (δ достаточно мал, чтобы потеря качества была небольшой).
- Обнаружим при доступе к функции разбиения словаря.
- Неустойчив к paraphrase — переписывание текста разрушает сигнал.

SynthID-text open-sourced в октябре 2024 через Google's Responsible GenAI Toolkit.

### Stable Signature (image)

Fernandez et al. ICCV 2023. Дообучить decoder латентной диффузии так, чтобы каждое сгенерированное изображение содержало фиксированное бинарное сообщение, встроенное в латентное представление. Обнаружение декодируется из latent с помощью neural decoder. Обрезанные (до 10% content) изображения обнаруживаются >90% при FPR<1e-6.

Май 2024 "Stable Signature is Unstable" (arXiv:2405.07145): fine-tuning decoder удаляет водяной знак, сохраняя качество изображения. Состязательное post-generation fine-tuning дешево; состязательная устойчивость водяного знака ограничена.

### Единый detector SynthID (ноябрь 2025)

Вместе с Gemini 3 Pro: multi-media detector, который считывает сигналы SynthID из текста, изображения, аудио и видео в одном API. Унифицирует стек provenance Google.

### C2PA

Coalition for Content Provenance and Authenticity. Криптографически подписанный tamper-evident стандарт metadata. C2PA 2.2 Explainer (2025). C2PA manifest записывает provenance claims (кто создал, когда, какие преобразования), подписанные ключом создателя.

Дополняет watermarking:
- Metadata можно удалить; водяные знаки — нет (легко).
- Metadata богата (полная цепочка provenance); водяные знаки несут биты.
- C2PA зависит от принятия платформами; водяные знаки встраиваются автоматически.

Google интегрирует оба в Search, Ads и "About this image."

### Ограничения

- **Model-specific.** Водяные знаки SynthID помечают генерации SynthID-enabled models. Генерация модели без SynthID не имеет водяного знака, поэтому "нет сигнала SynthID" не является доказательством подлинности.
- **Paraphrase.** Текстовые водяные знаки не переживают meaning-preserving paraphrase.
- **Transformation attacks.** arXiv:2508.20228 (2025) показывает meaning-preserving attacks, разрушающие и текстовые водяные знаки, и многие водяные знаки изображений.
- **Fine-tune removal.** Согласно "Stable Signature is Unstable," post-generation fine-tuning удаляет встроенные водяные знаки.

### EU AI Act Article 50

Transparency Code для маркировки AI-generated content (первый черновик декабрь 2025, второй черновик март 2026, ожидаемый финал июнь 2026 согласно [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)). Code остается черновиком по состоянию на апрель 2026, а сроки могут измениться. Регуляторный слой, который требует технический слой. Deepfakes должны маркироваться.

### Как это вписывается в Phase 18

Lessons 22-23 — о том, что модель испускает (приватные данные, сигнал provenance). Lesson 27 покрывает управление обучающими данными. Lesson 24 — регуляторная рамка, которая требует этих технических мер.

## Применение

`code/main.py` строит игрушечный текстовый водяной знак. Токены — целые числа 0..N-1; watermarked sampling смещается к hash-defined green set. Detector вычисляет green-token z-score. Можно наблюдать обнаружение на генерациях в 1000 токенов, увидеть, как paraphrase разрушает сигнал, и измерить false-positive rate на человеческом тексте.

## Результат

Этот урок создает `outputs/skill-provenance-audit.md`. Для заданного content deployment с provenance claim он аудирует: механизм водяного знака (если есть), цепочку подписи C2PA (если есть), состязательную устойчивость каждого компонента и покрытие по модальностям.

## Упражнения

1. Запустите `code/main.py`. Сообщите z-scores для watermarked 1000-token generation против human-authored text. Определите false-positive rate на пороге 95% confidence.

2. Реализуйте paraphrase attack, которая заменяет 30% токенов синонимами. Заново измерьте z-score.

3. Прочитайте Kirchenbauer et al. 2023 Section 6 об устойчивости. Почему текстовые водяные знаки ломаются при paraphrase, а водяные знаки изображений переживают обрезку?

4. Спроектируйте развертывание, использующее SynthID-text + C2PA metadata. Опишите цепочку provenance, которую видит потребитель. Укажите один режим отказа каждого компонента.

5. Результат 2024 года "Stable Signature is Unstable" показывает, что fine-tuning удаляет водяной знак изображения. Спроектируйте deployment control, который ограничивает эту атаку, например требование signed releases of fine-tuned checkpoints.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| SynthID | "Google's watermark" | Cross-modal сигнал provenance; текст, изображение, аудио, видео |
| Token watermark | "Kirchenbauer-style" | Текстовый водяной знак с biased-sampling, обнаружимый через green-token z-score |
| Stable Signature | "image watermark" | Водяной знак через fine-tuned-decoder; ICCV 2023 |
| C2PA | "the metadata standard" | Криптографически подписанная tamper-evident provenance metadata |
| Paraphrase robustness | "does rewording break it" | Свойство текстового водяного знака; сейчас ограничено |
| Fine-tune removal | "adversarial unwatermark" | Атака, удаляющая водяной знак изображения через decoder fine-tuning |
| Cross-modal detector | "unified SynthID" | Единый API ноября 2025 для разных модальностей |

## Дополнительное чтение

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — механизм token-watermark
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — статья о водяных знаках изображений
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — атака удаления
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — cross-modal watermark
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — стандарт metadata
