# Аудиоклассификация — от k-NN на MFCC до AST и BEATs

> Все от «лай собаки против сирены» до «какой это язык» — аудиоклассификация. Признаки — mels. Архитектура меняется каждое десятилетие. Оценка остается AUC, F1 и recall по классам.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 02 (спектрограммы и Mel), Фаза 3 · 06 (CNN), Фаза 5 · 08 (CNN и RNN для текста)
**Время:** ~75 минут

## Проблема

У вас есть 10-секундный клип. Вы хотите знать: «что это?» Городской звук (сирена, дрель, собака), речевая команда (yes/no/stop), идентификация языка (en/es/ar), эмоция говорящего (angry/neutral) или звук окружения (indoor/outdoor, babble). Все это — *аудиоклассификация*, и в 2026 году базовая архитектура зрелая: log-mel → CNN или Transformer → softmax.

Главная сложность — не сеть. Это данные. В аудиодатасетах жесткий дисбаланс классов, сильный domain shift (чистое против шумного) и шумные метки (кто решил, что это «городской гул», а не «шум ресторана»?). 80% задачи — кураторство, аугментация и оценка, а не замена CNN на Transformer.

## Концепция

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**k-NN на MFCC (бейзлайн 1990-х).** Сплющите MFCC по клипу, посчитайте cosine similarity к размеченному банку и верните majority vote по top K. Неожиданно силен на чистых маленьких датасетах (Speech Commands, ESC-50). Работает без GPU.

**2D CNN на log-mels (2015-2019).** Рассматривайте `(T, n_mels)` log-mel как изображение. Примените ResNet-18 или VGG-style. Сделайте global mean pool по временной оси. Softmax по классам. Это все еще бейзлайн в большинстве kaggle-соревнований 2026 года.

**Audio Spectrogram Transformer, AST (2021-2024).** Разбейте log-mel на патчи (например, 16×16), добавьте position embeddings, подайте в ViT. State of the art на AudioSet (mAP 0.485) для supervised learning.

**BEATs и WavLM-base (2024-2026).** Self-supervised pretraining на миллионах часов. Дообучение под вашу задачу требует 1-10% тех supervised данных, которые понадобились бы раньше. В 2026 году это дефолтная стартовая точка для неречевого аудио. BEATs-iter3 обходит AST на 1-2 mAP на AudioSet при четверти compute.

**Whisper-encoder как замороженный backbone (2024).** Возьмите encoder Whisper, уберите decoder, подключите линейный классификатор. Почти SOTA на language ID и простой классификации событий без аудиоаугментации. Бейзлайн «бесплатного обеда».

### Дисбаланс классов — настоящая проблема

ESC-50: 50 классов, по 40 клипов — сбалансированно, легко. UrbanSound8K: 10 классов, дисбаланс 10:1. AudioSet: 632 класса с long tail 100,000:1. Рабочие техники:

- Balanced sampling при обучении (не при оценке).
- Mixup: линейно интерполируйте два клипа и их метки как аугментацию.
- SpecAugment: маскируйте случайные полосы времени и частоты. Просто и критично.

### Оценка

- Взаимоисключающая multiclass (Speech Commands): top-1 accuracy, top-5 accuracy.
- Multiclass multi-label (AudioSet, UrbanSound-style): mean average precision (mAP).
- Сильный дисбаланс: recall по классам + macro F1.

Числа 2026 года, которые нужно знать:

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## Соберите это

### Шаг 1: извлеките признаки

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Шаг 2: фиксированное summary

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Просто, но сильно: mean + variance по времени дают 26-мерный фиксированный эмбеддинг для 13-coef MFCC. Работает мгновенно. Побеждало state-of-the-art NN-бейзлайны на ESC-50 еще в 2017 году.

### Шаг 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Шаг 4: перейдите к CNN на log-mels

В PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M параметров. Обучается за ~10 min на ESC-50 на одной RTX 4090. Точность 80%+.

### Шаг 5: дефолт 2026 года — дообучите BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Для BEATs используйте `microsoft/BEATs-base` через библиотеку `beats`; transformers API имеет ту же форму.

## Используйте это

Стек 2026 года:

| Ситуация | Начните с |
|-----------|-----------|
| Маленький датасет (<1000 клипов) | k-NN на средних MFCC (ваш бейзлайн) + аудиоаугментация |
| Средний датасет (1K–100K) | Дообучение BEATs или AST |
| Большой датасет (>100K) | Обучение с нуля или дообучение Whisper-encoder |
| Real-time, edge | 40-MFCC CNN, квантованный в int8 (KWS-style) |
| Multi-label (AudioSet) | BEATs-iter3 с BCE loss + mixup + SpecAugment |
| Language ID | MMS-LID, бейзлайн SpeechBrain VoxLingua107 |

Правило решения: **начинайте с замороженного backbone, а не со свежей модели**. Дообучение головы BEATs дает 95% SOTA за часы, а не недели.

## Доведите до результата

Сохраните как `outputs/skill-classifier-designer.md`. Выберите архитектуру, аугментации, стратегию баланса классов и метрику оценки для заданной задачи аудиоклассификации.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он обучает k-NN MFCC-бейзлайн на синтетическом датасете из 4 классов (чистые тоны разной высоты). Сообщите confusion matrix.
2. **Средне.** Замените `summarize` на [mean, var, skew, kurtosis]. Побеждает ли 4-moment pooling mean+var на том же синтетическом датасете?
3. **Сложно.** Используя `torchaudio`, обучите 2D CNN на ESC-50 fold 1. Сообщите accuracy для 5-fold cross-validation. Добавьте SpecAugment (time mask = 20, freq mask = 10) и сообщите delta.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| AudioSet | ImageNet для аудио | Датасет Google: 2M клипов, 632 класса, слабые метки YouTube. |
| ESC-50 | Маленький benchmark классификации | 50 классов × 40 клипов звуков окружения. |
| AST | Audio Spectrogram Transformer | ViT на log-mel патчах; SOTA 2021 года. |
| BEATs | Self-supervised аудио | Модель Microsoft, iter3 лидирует AudioSet на 2026 год. |
| Mixup | Парная аугментация | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Аугментация масками | Обнуление случайных временных и частотных полос спектрограммы. |
| mAP | Главная multi-label метрика | Mean average precision по классам и порогам. |

## Дополнительное чтение

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — эталонная архитектура 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — дефолт 2024+.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — доминирующая аудиоаугментация.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — живущий benchmark на 50 классов.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — таксономия YouTube на 632 класса; все еще золотой стандарт.
