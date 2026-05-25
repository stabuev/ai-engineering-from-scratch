# Генерация музыки — MusicGen, Stable Audio, Suno и лицензионное землетрясение

> Генерация музыки в 2026: Suno v5 и Udio v4 доминируют коммерчески; MusicGen, Stable Audio Open и ACE-Step лидируют в open-source. Техническая проблема почти решена. Юридическая проблема (settlement Warner Music на $500M, settlement UMG) перестроила область в 2025-2026.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 02 (спектрограммы), Фаза 4 · 10 (Diffusion Models)
**Время:** ~75 минут

## Проблема

Text → музыкальный клип от 30 секунд до 4 минут, с lyrics, vocals и structure. Три подзадачи:

1. **Instrumental generation.** Текст вроде «lo-fi hip-hop drums with warm keys» → audio. MusicGen, Stable Audio, AudioLDM.
2. **Song generation (with vocals + lyrics).** «Country song about rainy Texas nights» → полная песня. Suno, Udio, YuE, ACE-Step.
3. **Conditional / controllable.** Продлить существующий clip, перегенерировать bridge, сменить genre, stem-separate или inpaint. Inpainting + stem separation у Udio — функция 2026 года, которую нужно догонять.

## Концепция

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Token LM поверх neural-codec tokens

**MusicGen** от Meta (2023, MIT) и многие производные: condition on text/melody embeddings, autoregressively predict EnCodec tokens (32 kHz, 4 codebooks), decode with EnCodec. 300M - 3.3B params. Сильный бейзлайн; испытывает трудности после 30 секунд.

**ACE-Step** (open-source, 4B XL выпущен в апреле 2026) расширяет это до full-song lyric-conditioned generation. Ближайший open community аналог Suno.

### Diffusion по mels или latents

**Stable Audio (2023)** и **Stable Audio Open (2024)**: latent diffusion на compressed audio. Сильны в loops, sound design, ambient textures. Слабы в structured full songs.

**AudioLDM / AudioLDM2**: text-to-audio через T2I-style latent diffusion, обобщенный на music, sound effects, speech.

### Hybrid (production) — Suno, Udio, Lyria

Закрытые веса. Вероятно, AR codec LM + diffusion-based vocoder со specialized voice / drum / melody heads. Suno v5 (2026) — лидер качества ELO 1293. Udio v4 добавляет inpainting + stem separation (bass, drums, vocals отдельными downloads).

### Оценка

- **FAD (Fréchet Audio Distance).** Расстояние на уровне embeddings между generated и real audio distribution с VGGish или PANNs features. Ниже лучше. MusicGen small: 4.5 FAD на MusicCaps; SOTA ~3.0.
- **Musicality (subjective).** Человеческие предпочтения. Suno v5 ELO 1293 лидирует.
- **Text-audio alignment.** CLAP score между prompt и output.
- **Артефакты музыкальности.** Сбитые переходы, дрейф вокальных фраз, потеря структуры после 30 s.

## Карта моделей 2026 года

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## Юридический ландшафт (2025-2026)

- **Warner Music vs Suno settlement.** $500M. WMG теперь контролирует AI-likeness, music rights и user-generated tracks на Suno. Похожий UMG settlement по Udio.
- **EU AI Act** + **California SB 942**: AI-generated music должна маркироваться.
- **Riffusion / MusicGen** под MIT не несут compliance baggage, но и не дают коммерческих vocals.

Паттерны, пригодные к выпуску:

1. Генерировать только instrumental (MusicGen, Stable Audio Open, MIT/CC0 outputs).
2. Использовать commercial APIs (Suno, Udio, ElevenLabs Music) с per-generation license.
3. Обучаться на owned или licensed catalog (к этому приходит большинство enterprises).
4. Маркировать generations watermarks + metadata.

## Соберите это

### Шаг 1: сгенерируйте через MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Три размера: `small` (300M, fast), `medium` (1.5B), `large` (3.3B). Small достаточно, чтобы понять, «попала ли идея».

### Шаг 2: melody conditioning

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody принимает chromagram и сохраняет мелодию, меняя timbre. Полезно для «сделай эту мелодию струнным квартетом».

### Шаг 3: оценка FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Считает VGGish-embedding distance. Полезно для genre-level regression tests; не заменяет human listeners.

### Шаг 4: добавление в LLM-music workflow

Объедините с идеями из Lessons 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Используйте это

| Цель | Стек |
|------|-------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime (closed) |
| Full songs with vocals (commercial) | Suno v5 или Udio v4 с explicit license |
| Full songs with vocals (open) | ACE-Step XL или YuE |
| Short ad jingle | MusicGen melody-conditioned на hummed reference |
| Music-video background | MusicGen + Stable Video Diffusion |

## Ловушки, которые все еще попадают в продакшен в 2026 году

- **Copyright-laundering prompts.** "Song in the style of Taylor Swift" — commercial Suno/Udio теперь это фильтруют, open models нет. Добавьте свой filter list.
- **Repetition / drift past 30 s.** AR models зацикливаются. Склеивайте несколько generations через crossfade или используйте ACE-Step для structural coherence.
- **Tempo drift.** Модели уходят от BPM. Используйте BPM tags в prompt и post-filter через `beat_track` из librosa.
- **Vocal intelligibility.** Suno отличен; open models часто мажут слова. Если lyrics важны, используйте commercial API или fine-tune.
- **Mono output.** Open models генерируют mono или fake-stereo. Улучшайте proper stereo reconstruction (ezst, stereo diffusion Cartesia).

## Доведите до результата

Сохраните как `outputs/skill-music-designer.md`. Выберите модель, стратегию лицензирования, план длины/структуры и disclosure metadata для music-gen deployment.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он создает "generative" chord progression + drum pattern как ASCII symbols — карикатуру на music-gen. При желании воспроизведите через любой MIDI renderer.
2. **Средне.** Установите `audiocraft`, сгенерируйте 10-секундные clips по 4 genre prompts с MusicGen-small, измерьте FAD против reference genre set.
3. **Сложно.** Используя ACE-Step (или MusicGen-melody), сгенерируйте три вариации одной мелодии с разными timbre prompts. Посчитайте CLAP similarity к prompt, чтобы проверить alignment.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| FAD | Audio FID | Fréchet distance между embedding distributions real vs generated. |
| Chromagram | Melody как pitches | 12-dim per-frame vector; вход для melody conditioning. |
| Stems | Instrument tracks | Разделенные bass / drums / vocals / melody как WAV. |
| Inpainting | Перегенерация секции | Mask time window; model regenerates just that. |
| CLAP | Text-audio CLIP | Contrastive audio-text embedding; eval text-audio alignment. |
| EnCodec | Music codec | Neural codec Meta, используемый MusicGen; 32 kHz, 4 codebooks. |

## Дополнительное чтение

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — open autoregressive benchmark.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — дефолт для sound design.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — open 4B full-song generator, April 2026.
- [Suno v5 platform docs](https://suno.com) — коммерческий лидер качества.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latent diffusion для music + sound effects.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — прецедент Nov 2025.
