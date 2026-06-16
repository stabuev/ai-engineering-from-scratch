# Распознавание и верификация говорящего

> ASR спрашивает: «что сказали?» Распознавание говорящего спрашивает: «кто это сказал?» Математика выглядит так же — эмбеддинги плюс cosine — но каждое продакшен-решение упирается в одно число EER.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 02 (спектрограммы и Mel), Фаза 5 · 22 (Embedding Models)
**Время:** ~45 минут

## Цели обучения

- Строить эмбеддинги диктора и оценивать верификацию косинусной близостью и порогом.
- Вычислять EER из пар сходства и объяснять, почему это ключевое продакшен-число.
- Различать верификацию, идентификацию и open-set и размещать диаризацию.

## Проблема

Пользователь произносит passphrase. Вы хотите понять: это тот человек, за которого он себя выдает (*verification*, 1:1), или это первый человек в вашем enrollment bank (*identification*, 1:N)? Или ни то ни другое — это неизвестный говорящий (*open-set*)?

До 2018: GMM-UBM + i-vectors. Приемлемый EER, но хрупкость к сдвигу канала (телефон против ноутбука) и эмоциям. 2018–2022: x-vectors (TDNN backbone, обученный с angular margin). 2022+: ECAPA-TDNN и WavLM-large embeddings. К 2026 году область держится на трех моделях и одной метрике.

Метрика — **EER**, Equal Error Rate. Настройте threshold решения так, чтобы False Accept Rate = False Reject Rate. Точка пересечения — EER. Используется в каждой статье, каждом leaderboard и каждом закупочном обсуждении.

## Концепция

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Пайплайн.** Enrollment: запишите 5–30 секунд целевого говорящего; вычислите эмбеддинг фиксированной размерности (192-d для ECAPA-TDNN, 256-d для WavLM-large). Verification: получите эмбеддинг тестового высказывания; посчитайте cosine similarity; сравните с threshold.

**ECAPA-TDNN (2020, все еще доминирует в 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. 1D conv blocks с squeeze-excitation, multi-head attention pooling, затем linear layer в 192-d. Обучается на VoxCeleb 1+2 (2,700 speakers, 1.1M utterances) с Additive Angular Margin loss (AAM-softmax).

**WavLM-SV (2022+).** Дообучите pretrained WavLM-large SSL backbone с AAM loss. Качество выше, но медленнее — 300+ MB против 15 MB.

**x-vector (бейзлайн).** TDNN + statistics pooling. Классика; все еще полезна на CPU / edge.

**AAM-softmax.** Обычный softmax с добавленным margin `m` в angular space: `cos(θ + m)` для правильного класса. Принуждает межклассовое угловое разделение. Типично `m=0.2`, scale `s=30`.

### Скоринг

- **Cosine** между enrollment и test embeddings. Решение по threshold.
- **PLDA (Probabilistic LDA).** Проецирует embeddings в latent space, где same-speaker vs different-speaker имеет closed-form likelihood ratio. Добавляется поверх cosine для снижения EER на +10–20%. Стандарт до 2020; теперь только для closed-set setups.
- **Нормализация score.** `S-norm` или `AS-norm`: нормализует каждый score относительно cohort of imposter means and stds. Критично для cross-domain eval.

### Числа, которые нужно знать (2026)

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Диаризация

"Who spoke when" в многоговорящем клипе. Пайплайн: VAD → segment → embed each segment → cluster (agglomerative или spectral) → smooth boundaries. Современный стек: `pyannote.audio` 3.1, где speaker segmentation + embedding + clustering спрятаны за одним вызовом. SOTA DER 2026 года на AMI — ~15% (против 23% в 2022).

## Соберите это

### Шаг 1: игрушечный эмбеддинг из статистик MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Совсем не SOTA — только для обучения. `code/main.py` использует это как proof-of-concept на синтетических данных говорящих.

### Шаг 2: cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Шаг 3: EER по similarity pairs

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Возвращает (eer, threshold_at_eer). Сообщайте оба.

### Шаг 4: продакшен со SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Шаг 5: диаризуйте с pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deepfake detection) | AASIST или RawNet2 |
| Tiny embedded (KWS + enrollment) | Titanet-Small (NeMo) |

## Ловушки

- **Channel mismatch.** Модель обучена на VoxCeleb (web video) ≠ phone-call audio. Всегда оценивайте на целевом канале.
- **Короткие utterances.** EER резко ухудшается ниже 3 секунд test audio.
- **Enrollment с шумом.** Один шумный enrollment портит anchor. Используйте ≥3 чистых samples и усредняйте.
- **Фиксированный threshold между условиями.** Всегда настраивайте threshold на held-out dev set из целевого домена.
- **Cosine на ненормализованных embeddings.** Сначала L2-normalize; иначе доминирует magnitude.

## Доведите до результата

Сохраните как `outputs/skill-speaker-verifier.md`. Выберите модель, протокол enrollment, план настройки threshold и защиту от мошенничества.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он строит синтетических "speakers" (разные tone profiles), делает enrollment и считает EER на trial list из 100 пар.
2. **Средне.** Используйте SpeechBrain ECAPA на 30 VoxCeleb1 utterances (5 speakers × 6 each). Посчитайте EER с cosine против PLDA.
3. **Сложно.** Соберите полный пайплайн enroll → diarize → verify с `pyannote.audio`. Оцените DER на AMI dev set.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| EER | Главная метрика | Threshold, где False Accept = False Reject. |
| Verification | 1:1 | "Is this Alice?" |
| Identification | 1:N | "Who is speaking?" |
| Open-set | Возможен unknown | Test set может содержать незарегистрированных speakers. |
| Enrollment | Регистрация | Вычисление reference embedding говорящего. |
| AAM-softmax | Loss | Softmax с additive angular margin; заставляет кластеры разделяться. |
| PLDA | Классический scoring | Probabilistic LDA; likelihood-ratio scoring поверх embeddings. |
| DER | Метрика диаризации | Diarization Error Rate — miss + false alarm + confusion. |

## Дополнительное чтение

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — классическая статья о deep embeddings.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — доминирующая архитектура 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SSL backbone для SV и diarization.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — продакшен-стек diarization + embedding.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — текущие EER-позиции моделей.
