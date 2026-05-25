# Voice Anti-Spoofing и аудиоводяные знаки — ASVspoof 5, AudioSeal, WaveVerify

> Voice cloning развивался быстрее защит. Продакшен-голосовые системы 2026 года требуют две вещи: detector (AASIST, RawNet2), отличающий настоящую речь от синтетической, и watermark (AudioSeal), переживающий сжатие и редактирование. Поставляйте оба компонента или не поставляйте voice cloning.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 06 (Speaker Recognition), Фаза 6 · 08 (Voice Cloning)
**Время:** ~75 минут

## Проблема

Три связанные защиты:

1. **Anti-spoofing / deepfake detection.** По аудиоклипу определить, синтетический он или настоящий. Бенчмарки ASVspoof (ASVspoof 2019 → 2021 → 5) — золотой стандарт.
2. **Audio watermarking.** Встроить в сгенерированное аудио незаметный для слуха сигнал, который detector сможет извлечь позже. AudioSeal (Meta) и WavMark — открытые варианты.
3. **Authenticated provenance.** Криптографическая подпись аудиофайлов и метаданных. C2PA / Content Authenticity Initiative.

Detection работает против противников, которые не сотрудничают. Watermarking обеспечивает compliance: AI-generated audio должно быть распознаваемо как таковое. В 2026 нужны оба.

## Концепция

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — benchmark 2024-2025

Главное отличие от прошлых выпусков:

- **Crowdsourced data** (не студийно-чистые записи) — реалистичные условия.
- **~2000 speakers** (против ~100 раньше).
- **32 attack algorithms.** TTS + voice conversion + adversarial perturbation.
- **Two tracks.** Countermeasure (CM) как отдельная детекция; Spoofing-robust ASV (SASV) для биометрических систем.

State-of-the-art на ASVspoof 5: ~7.23% EER. На старом ASVspoof 2019 LA: 0.42% EER. В реальном deployment ждите 5-10% EER на клипах из реальной среды.

### AASIST и RawNet2 — семейства моделей детекции

**AASIST** (2021, обновляется до 2026). Graph-attention на спектральных признаках. Текущий SOTA для задачи countermeasure в ASVspoof 5.

**RawNet2.** Сверточный front-end поверх raw waveform + TDNN backbone. Более простой baseline; все еще конкурентоспособен при fine-tuning.

**NeXt-TDNN + SSL features.** Вариант 2025 года: ECAPA-style + признаки WavLM + focal loss. Дает 0.42% EER на ASVspoof 2019 LA.

### AudioSeal — дефолтный watermark 2024 года

**AudioSeal** от Meta (Jan 2024, v0.2 Dec 2024). Дизайн:

- **Localized.** Детектирует watermark по фреймам с разрешением отсчетов 16 kHz (1/16000 s).
- **Generator + detector jointly trained.** Generator учится встраивать неслышимый сигнал; detector ищет его после аугментаций.
- **Robust.** Выживает после MP3 / AAC compression, EQ, speed-shift ±10%, noise mix +10 dB SNR.
- **Fast.** Detector работает на 485× realtime; в 1000 раз быстрее WavMark.
- **Capacity.** 16-bit payload (model ID, generation timestamp, user ID) можно встроить в каждое высказывание.

### WavMark

Открытый baseline до AudioSeal. Invertible neural network, 32 bits/sec. Проблемы:

- Синхронизация через brute force медленная.
- Удаляется Gaussian noise или MP3 compression.
- Плохо подходит для real-time.

### WaveVerify (July 2025)

Закрывает слабости AudioSeal — особенно temporal manipulations (reversal, speed). Использует FiLM-based generator + Mixture-of-Experts detector. Конкурирует с AudioSeal на стандартных атаках и обрабатывает temporal edits.

### Разрыв, которым пользуются adversaries

Из AudioMarkBench: «при pitch shift все watermarks показывают Bit Recovery Accuracy ниже 0.6, что указывает на почти полное удаление». **Pitch-shift — универсальная атака.** Ни один watermark 2026 не полностью устойчив к агрессивному изменению высоты тона. Поэтому detection (AASIST) нужен вместе с watermarking.

### C2PA / Content Authenticity Initiative

Не ML-техника, а формат manifest. Аудиофайлы несут криптографически подписанные metadata об инструменте создания, авторе и дате. Audobox / Seamless используют это. Хорошо для provenance; не помогает, если злоумышленник перекодирует файл и удалит metadata.

## Соберите это

### Шаг 1: простой detector на спектральных признаках (toy)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

У синтетической речи часто необычно плоская высокочастотная энергия. В продакшене используют AASIST, а не такой detector. Но интуиция сохраняется.

### Шаг 2: AudioSeal embed + detect

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Шаг 3: оценка — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Шаг 4: интеграция в продакшен

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Каждая генерация поставляется с: (1) watermark, (2) подписанным manifest, (3) audit log, соответствующим политике хранения.

## Используйте это

| Сценарий | Защита |
|----------|---------|
| Поставка TTS / voice cloning | Встраивать AudioSeal в каждый output, без исключений |
| Биометрическая разблокировка голосом | AASIST + ECAPA ensemble; liveness challenge |
| Детекция мошенничества в call-center | AASIST на 20% выборке входящих звонков |
| Подлинность подкастов | C2PA signing при upload, AudioSeal для AI-generated |
| Исследования / обучение detectors | ASVspoof 5 train/dev/eval sets |

## Ловушки

- **Watermark без запуска detector.** Бессмысленно. Поставляйте detector в своем CI.
- **Detection без калибровки.** AASIST, обученный на ASVspoof LA, переобучается; точность в реальном мире падает. Калибруйте на своем домене.
- **Pitch-shift gap.** Агрессивный pitch shift удаляет большинство watermarks. Нужен запасной detection.
- **Metadata strip-and-rehost.** C2PA тривиально обходится перекодированием. Всегда сочетайте криптографическую и перцептивную защиту (watermark).
- **Liveness как detection.** Попросить случайную фразу. Предотвращает replay attacks, но не real-time cloning.

## Доведите до результата

Сохраните как `outputs/skill-spoof-defender.md`. Выберите detection model, watermark, provenance manifest и operational playbook для deployment voice generation.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Toy detector + toy watermark embed/detect на синтетическом audio.
2. **Средне.** Установите `audioseal`, встройте 16-bit payload в TTS output, затем декодируйте заново. Испортите аудио шумом и измерьте Bit Recovery Accuracy.
3. **Сложно.** Дообучите RawNet2 или AASIST на ASVspoof 2019 LA. Измерьте EER. Протестируйте на held-out наборе клипов, сгенерированных F5-TTS, и посмотрите, как деградирует OOD detection.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| ASVspoof | Benchmark | Двухгодичный challenge; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detector | Классификатор: настоящая речь против синтетической / преобразованной. |
| SASV | Speaker verif + CM | Интегрированная биометрия + spoof detection. |
| AudioSeal | Meta watermark | Локализованный, 16-bit payload, в 485× быстрее WavMark. |
| Bit Recovery Accuracy | Живучесть watermark | Доля битов payload, восстановленных после атаки. |
| C2PA | Provenance manifest | Криптографические metadata о создании / авторстве. |
| AASIST | Семейство detectors | Graph-attention-based SOTA для anti-spoofing. |

## Дополнительное чтение

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — текущий benchmark.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — дефолтный watermark.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE detector для temporal attacks.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA backbone для detection.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — оценка устойчивости.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — формат provenance manifest.
