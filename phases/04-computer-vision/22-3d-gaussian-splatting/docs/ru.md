# 3D Gaussian Splatting с нуля

> Сцена - это облако из миллионов 3D-гауссиан. У каждой есть положение, ориентация, масштаб, непрозрачность и цвет, зависящий от направления взгляда. Растеризуйте их, выполните обратное распространение через растеризацию - готово.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 4 Урок 13 (3D Vision & NeRF), Фаза 1 Урок 12 (Tensor Operations), Фаза 4 Урок 10 (Diffusion basics optional)
**Время:** ~90 минут

## Цели обучения

- Объяснить, почему 3D Gaussian Splatting заменил NeRF как производственный стандарт по умолчанию для фотореалистичной 3D-реконструкции в 2026 году
- Назвать шесть параметров каждой гауссианы (положение, кватернион вращения, масштаб, непрозрачность, цвет через сферические гармоники, необязательный признак) и сколько чисел с плавающей точкой дает каждый из них
- Реализовать с нуля 2D-растеризатор Gaussian splatting с `alpha`-композитингом, затем показать, как 3D-случай проецируется в тот же цикл
- Использовать `nerfstudio`, `gsplat` или `SuperSplat`, чтобы реконструировать сцену по 20-50 фотографиям и экспортировать ее в расширение glTF `KHR_gaussian_splatting` или схему OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat`

## Проблема

NeRF хранит сцену как веса MLP. Каждый отрендеренный пиксель - это сотни запросов к MLP вдоль луча. Обучение занимает часы, рендеринг занимает секунды, а веса нельзя редактировать - если вы хотите передвинуть стул внутри сцены, придется переобучать модель.

3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023) заменил все это. Сцена - это явный набор 3D-гауссиан. Рендеринг - GPU-растеризация со скоростью 100+ fps. Обучение занимает минуты. Редактирование прямое: сдвиньте подмножество гауссиан, и стул переместится. К 2026 году Khronos Group ратифицировала расширение glTF для Gaussian splats, OpenUSD 26.03 поставляется со схемой Gaussian splat, Zillow и Apartments.com рендерят с их помощью недвижимость, а большинство новых исследовательских работ по 3D-реконструкции являются вариантами базовой идеи 3DGS.

Ментальная модель проста, но в математике достаточно движущихся частей, поэтому большинство введений начинают с растеризации и пропускают проекции и сферические гармоники. Этот урок строит всю систему целиком - сначала 2D-версию, затем 3D-расширение.

## Концепция

### Что несет гауссиана

Одна 3D-гауссиана - это параметрическое пятно в пространстве со следующими атрибутами:

```
position         mu         (3,)    centre in world coordinates
rotation         q          (4,)    unit quaternion encoding orientation
scale            s          (3,)    log-scales per axis (exponentiated at render time)
opacity          alpha      (1,)    post-sigmoid opacity [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   view-dependent colour
```

Вращение + масштаб строят ковариацию 3x3: `Sigma = R S S^T R^T`. Это форма гауссианы в 3D. Сферические гармоники позволяют цвету меняться в зависимости от направления взгляда - зеркальные блики, тонкий блеск, зависящее от вида свечение - без хранения текстур для каждого вида. При степени SH 3 получается 16 коэффициентов на цветовой канал, 48 чисел с плавающей точкой на одну гауссиану только для цвета.

Сцена обычно содержит 1-5 миллионов гауссиан. Каждая хранит примерно 60 чисел с плавающей точкой (3 + 4 + 3 + 1 + 48 + прочее). Это 240 MB для сцены из пяти миллионов гауссиан - намного меньше, чем эквивалентное облако точек с текстурой на каждую точку, и на порядок меньше, чем веса MLP у NeRF при повторном рендеринге в высоком разрешении.

### Растеризация, а не ray marching

```mermaid
flowchart LR
    SCENE["Millions of 3D Gaussians<br/>(position, rotation, scale,<br/>opacity, SH colour)"] --> PROJ["Project to 2D<br/>(camera extrinsics + intrinsics)"]
    PROJ --> TILES["Assign to tiles<br/>(16x16 screen-space)"]
    TILES --> SORT["Depth-sort<br/>per tile"]
    SORT --> ALPHA["Alpha-composite<br/>front-to-back"]
    ALPHA --> PIX["Pixel colour"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

Пять шагов, и все удобны для GPU. Никакого запроса к MLP для каждого пикселя. Одна RTX 3080 Ti рендерит 6 миллионов splats со скоростью 147 fps.

### Шаг проекции

3D-гауссиана в мировой позиции `mu` с 3D-ковариацией `Sigma` проецируется в 2D-гауссиану в экранной позиции `mu'` с 2D-ковариацией `Sigma'`:

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

Отпечаток 2D-гауссианы - это эллипс, оси которого являются собственными векторами `Sigma'`. Каждый пиксель внутри этого эллипса получает вклад гауссианы, взвешенный через `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`.

### Правило alpha-композитинга

Для одного пикселя гауссианы, которые его покрывают, сортируются от дальних к ближним (или, что эквивалентно, от ближних к дальним с инвертированной формулой). Цвет композитится тем же уравнением, что и в каждом полупрозрачном растеризаторе с 1980-х годов:

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour
```

Это **то же самое уравнение, что и объемный рендеринг NeRF**, только поверх явного разреженного набора гауссиан вместо плотных сэмплов вдоль луча. Именно поэтому качество рендеринга сопоставимо с NeRF - оба подхода интегрируют одно и то же уравнение поля излучения (radiance field).

### Почему это дифференцируемо

Каждый шаг - проекция, назначение тайлов, alpha-композитинг, вычисление SH - дифференцируем относительно параметров гауссианы. Имея эталонное изображение, вычислите пиксельную потерю рендера, выполните обратное распространение через растеризатор, обновите все `(mu, q, s, alpha, c_lm)` градиентным спуском. За ~30,000 итераций гауссианы находят правильные положения, масштабы и цвета.

### Уплотнение и отсечение

Фиксированный набор гауссиан не может покрыть сложную сцену. Обучение включает два адаптивных механизма:

- **Клонировать** гауссиану в ее текущей позиции, когда величина ее градиента высока, но масштаб мал - реконструкции здесь нужно больше деталей.
- **Разделить** крупномасштабную гауссиану на две меньшие, когда ее градиент высок - одна большая гауссиана слишком гладкая, чтобы подогнать область.
- **Отсечь** гауссианы, чья непрозрачность падает ниже порога - они не вносят вклада.

Уплотнение запускается каждые N итераций. Сцена обычно растет от ~100k начальных гауссиан (инициализированных из точек SfM) до 1-5M к концу обучения.

### Сферические гармоники в одном абзаце

Зависящий от вида цвет - это функция `c(direction)` на единичной сфере. Сферические гармоники - это базис Фурье на сфере. Обрежьте его на степени `L`, и получите `(L+1)^2` базисных функций на канал. Вычисление цвета для нового вида - это скалярное произведение между выученными SH-коэффициентами и базисом, вычисленным в направлении взгляда. Степень 0 = один коэффициент = постоянный цвет. Степень 3 = 16 коэффициентов = достаточно, чтобы захватить ламбертово затенение, зеркальность и мягкое отражение. Статьи по 3D Gaussian Splatting по умолчанию используют степень 3.

### Производственный стек 2026 года

```
1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D- и генеративные варианты

- **4D Gaussian Splatting** - гауссианы являются функциями времени; используется для объемного видео (Superman 2026, "Helicopter" A$AP Rocky).
- **Generative splats** - модели text-to-splat (Marble от World Labs), которые галлюцинируют целые сцены.
- **3D Gaussian Unscented Transform** - вариант NVIDIA NuRec для симуляции автономного вождения.

## Соберите это

### Шаг 1: 2D-гауссиана

Сначала построим 2D-растеризатор. 3D-случай сводится к нему после проекции.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` вычисляет квадратичную форму `diff^T Sigma^-1 diff` для каждой пары (гауссиана, пиксель).

### Шаг 2: 2D splatting-растеризатор

Alpha-композитинг от ближних к дальним. Глубина в 2D не имеет смысла, поэтому для порядка мы используем выучиваемый скаляр на каждую гауссиану.

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

Не быстро - реальная реализация использует CUDA-ядра на основе тайлов, - но это ровно правильная математика и полностью дифференцируемая схема.

### Шаг 3: Обучаемая 2D-сцена splats

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`, `opacity_logit` и `colour_logits` - все это неограниченные параметры, которые при рендеринге отображаются через правильную активацию. Это стандартный шаблон для каждой реализации 3DGS.

### Шаг 4: Подогнать 2D-гауссианы к целевому изображению

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

За 200 шагов 64 гауссианы сходятся к двум фигурам. Это вся идея - градиентный спуск по явным геометрическим примитивам.

### Шаг 5: От 2D к 3D

3D-расширение сохраняет тот же цикл. Добавления:

1. Вращение каждой гауссианы - это кватернион вместо одного угла.
2. Ковариация равна `R S S^T R^T`, где `R` строится из кватерниона, а `S = diag(exp(log_scale))`.
3. Проекция `(mu, Sigma) -> (mu', Sigma')` использует внешние параметры камеры и якобиан перспективной проекции в точке `mu`.
4. Цвет становится разложением по сферическим гармоникам; вычисляйте его в направлении взгляда.
5. Сортировка по глубине идет по настоящему z в пространстве камеры, а не по выученному скаляру.

Каждая производственная реализация (`gsplat`, `inria/gaussian-splatting`, `nerfstudio`) делает именно это на GPU с CUDA-ядрами на основе тайлов.

### Шаг 6: Вычисление сферических гармоник

Базис SH до степени 3 имеет 16 членов на канал. Вычисление:

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

Выученные `sh_coeffs` хранят "цвет во всех направлениях" для этой гауссианы. При рендеринге вы вычисляете их относительно текущего направления взгляда и получаете 3-вектор RGB.

## Используйте это

Для реальной работы с 3DGS используйте `gsplat` (Meta) или `nerfstudio`:

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` - это 3DGS-тренер в nerfstudio. Запуск занимает 10-30 минут на RTX 4090 для типичной сцены.

Важные варианты экспорта в 2026 году:

- `.ply` - сырое облако гауссиан (переносимое, самый большой файл).
- `.splat` - квантованный формат PlayCanvas / SuperSplat.
- glTF `KHR_gaussian_splatting` - стандарт Khronos, переносимый между просмотрщиками (RC февраля 2026 года).
- OpenUSD `UsdVolParticleField3DGaussianSplat` - нативный для USD, для пайплайнов NVIDIA Omniverse и Vision Pro.

Для 4D / динамических сцен `4DGS` и `Deformable-3DGS` расширяют тот же механизм меняющимися во времени средними и непрозрачностями.

## Отправьте в работу

Этот урок создает:

- `outputs/prompt-3dgs-capture-planner.md` - промпт, который планирует съемочную сессию (количество фотографий, траекторию камеры, освещение) для заданного типа сцены.
- `outputs/skill-3dgs-export-router.md` - навык, который выбирает правильный формат экспорта (`.ply` / `.splat` / glTF / USD) с учетом последующего просмотрщика или движка.

## Упражнения

1. **(Легко)** Запустите приведенный выше 2D splat-тренер на другом синтетическом изображении. Меняйте `num_splats` в `[16, 64, 256]` и постройте график MSE от шага для каждого значения. Найдите точку убывающей отдачи.
2. **(Средне)** Расширьте 2D-растеризатор, чтобы поддерживать RGB-цвета каждой гауссианы, зависящие от скалярного "угла взгляда" через гармонику степени 2. Обучите на паре целевых изображений и проверьте, что модель реконструирует оба.
3. **(Сложно)** Склонируйте `nerfstudio` и обучите `splatfacto` на съемке любой вашей сцены из 20 фотографий (стол, растение, лицо, комната). Экспортируйте в glTF `KHR_gaussian_splatting` и откройте в просмотрщике (Three.js `GaussianSplats3D`, SuperSplat, Babylon.js V9). Сообщите время обучения, количество гауссиан и fps рендера.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| 3DGS | "Гауссовы splats (Gaussian splats)" | Явное представление сцены как миллионов 3D-гауссиан с положением, вращением, масштабом, непрозрачностью и SH-цветом для каждой гауссианы |
| Ковариация (Covariance) | "Форма гауссианы (Shape of the Gaussian)" | `Sigma = R S S^T R^T`; ориентация и анизотропный масштаб одной гауссианы |
| Alpha-композитинг (Alpha compositing) | "Смешивание от дальних к ближним (Back-to-front blend)" | То же уравнение, что и объемный рендеринг NeRF, теперь поверх явного разреженного набора |
| Уплотнение (Densification) | "Клонировать и разделять (Clone and split)" | Адаптивное добавление новых гауссиан там, где реконструкция недообучена |
| Отсечение (Pruning) | "Удалять низкую непрозрачность (Delete low-opacity)" | Удаление гауссиан, которые во время обучения схлопнулись до почти нулевой непрозрачности |
| Сферические гармоники (Spherical harmonics) | "Зависящий от вида цвет (View-dependent colour)" | Базис Фурье на сфере; хранит цвет как функцию направления взгляда |
| Splatfacto | "3DGS в nerfstudio (nerfstudio's 3DGS)" | Самый простой путь к обучению 3DGS в 2026 году |
| `KHR_gaussian_splatting` | "Стандарт glTF (glTF standard)" | Расширение Khronos 2026 года, которое делает 3DGS переносимым между просмотрщиками и движками |

## Дополнительное чтение

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - оригинальная статья
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) - CUDA-растеризатор производственного качества
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) - эталонный рецепт обучения
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) - переносимый формат 2026 года
- [OpenUSD 26.03 release notes](https://openusd.org/release/) - схема `UsdVolParticleField3DGaussianSplat`
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) - отраслевой обзор
