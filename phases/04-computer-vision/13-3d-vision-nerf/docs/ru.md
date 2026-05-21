# 3D-зрение — облака точек и NeRF

> 3D-зрение бывает двух видов. Облака точек — это сырые выходные данные сенсора. NeRF — это выученное объемное поле. Оба отвечают на вопрос «что где находится в пространстве».

**Тип:** изучить + построить
**Языки:** Python
**Предварительные требования:** фаза 4, урок 03 (CNNs), фаза 1, урок 12 (Tensor Operations)
**Время:** ~45 минут

## Цели обучения

- Различать явные (облако точек, mesh, voxel) и неявные (signed distance field, NeRF) 3D-представления и понимать, когда используется каждое из них
- Понять трюк PointNet с симметричной функцией, который делает нейронную сеть инвариантной к перестановкам на неупорядоченном множестве точек
- Проследить прямой проход NeRF: трассировка лучей, объемный рендеринг, позиционное кодирование, MLP-голова для плотности+цвета
- Использовать `nerfstudio` или `instant-ngp` для предобученной 3D-реконструкции по небольшому набору изображений с известными позами

## Проблема

Камера создает 2D-изображение. LIDAR создает множество 3D-точек без порядка. Конвейер structure-from-motion создает разреженное облако 3D-ключевых точек. NeRF реконструирует целую 3D-сцену по нескольким изображениям с известными позами. Все это — «зрение», но ничто из этого не похоже на плотный тензор, который ожидает CNN.

3D-зрение важно, потому что почти каждая ценная робототехническая задача выполняется в 3D: захват объектов, обход препятствий, навигация, окклюзия в AR, захват 3D-контента. Инженер по компьютерному зрению, который понимает только 2D-изображения, отрезан от самого быстрорастущего сегмента области (AR/VR-контент, робототехника, стеки автономного вождения, 3D-реконструкция на основе NeRF для недвижимости или строительства).

Эти два представления доминируют по разным причинам. Облака точек — это то, что сенсоры дают вам бесплатно. NeRF и их преемники (3D Gaussian splatting, neural SDFs) — это то, что вы получаете, когда просите нейронную сеть выучить сцену.

## Концепция

### Облака точек

Облако точек — это неупорядоченное множество из N точек в R^3, где каждая точка опционально имеет признаки (цвет, интенсивность, нормаль).

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

Нет сетки, нет связности. Два свойства делают это сложным для нейронных сетей:

- **Инвариантность к перестановкам** — выход не должен зависеть от порядка точек.
- **Переменное N** — одна модель должна обрабатывать облака разных размеров.

PointNet (Qi et al., 2017) решил обе проблемы одной идеей: применить общий MLP к каждой точке, а затем агрегировать симметричной функцией (max pool). Результат — вектор фиксированного размера, который не зависит от порядка.

```
f(P) = max_{p in P} MLP(p)
```

Это все ядро PointNet. Более глубокие варианты (PointNet++, Point Transformer) добавляют иерархическую выборку и локальную агрегацию, но трюк с симметричной функцией остается тем же.

### Архитектура PointNet

```mermaid
flowchart LR
    PTS["N points<br/>(x, y, z)"] --> MLP1["shared MLP<br/>(64, 64)"]
    MLP1 --> MLP2["shared MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["max pool<br/>(symmetric)"]
    MAX --> FEAT["global feature<br/>(1024,)"]
    FEAT --> FC["MLP classifier"]
    FC --> CLS["class logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

"Shared MLP" означает, что один и тот же MLP запускается на каждой точке независимо. Для эффективности реализуется как 1x1 conv по измерению точек.

### Neural Radiance Fields (NeRFs)

NeRF (Mildenhall et al., 2020) взяли вопрос «можем ли мы реконструировать 3D-сцену по N фотографиям?» и ответили на него нейронной сетью, которая сама является сценой. Сеть отображает `(x, y, z, viewing_direction)` в `(density, colour)`. Рендеринг нового вида — это цикл трассировки лучей по этой сети.

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

Функция потерь сравнивает отрендеренный пиксель с истинным пикселем на обучающих фотографиях. Обратное распространение через шаг рендеринга обновляет MLP. Нет 3D-разметки, нет явной геометрии — сцена хранится в весах MLP.

### Позиционное кодирование в NeRF

Обычный MLP на `(x, y, z)` не может представлять высокочастотные детали, потому что MLP спектрально смещены в сторону низких частот. NeRF исправляет это, кодируя каждую координату в вектор признаков Фурье перед MLP:

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

До L=10 уровней частот. Это тот же трюк, который трансформеры используют для позиций, и он снова появляется в обусловливании времени в диффузии (урок 10). Без него NeRF выглядят размытыми.

### Объемный рендеринг

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i` — это пропускание (transmittance): сколько света доходит до точки i. `(1 - exp(-sigma_i * delta_i))` — это непрозрачность в точке i. `c_i` — цвет. Итоговый пиксель — взвешенная сумма вдоль луча.

### Что заменило NeRF

Чистые NeRF медленно обучаются (часы) и медленно рендерятся (секунды на изображение). Последующая линия развития:

- **Instant-NGP** (2022) — hash-grid encoding заменяет позиционный вход MLP; обучается за секунды.
- **Mip-NeRF 360** — обрабатывает неограниченные сцены и anti-aliasing.
- **3D Gaussian Splatting** (2023) — заменяет объемное поле миллионами 3D-гауссиан; обучается за минуты, рендерится в реальном времени. Текущий производственный стандарт.

Почти каждый реальный NeRF-продукт в 2026 году на самом деле использует 3D Gaussian splatting. Ментальная модель все еще остается NeRF.

### Датасеты и бенчмарки

- **ShapeNet** — классификация и сегментация 3D CAD-моделей как облаков точек.
- **ScanNet** — реальные сканы помещений для сегментации.
- **KITTI** — уличные облака точек LIDAR для автономного вождения.
- **NeRF Synthetic** / **Blended MVS** — датасеты изображений с позами для синтеза видов.
- **Mip-NeRF 360** dataset — неограниченные реальные сцены.

## Построим это

### Шаг 1: классификатор PointNet

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — transposed for Conv1d
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

Около 1.6M параметров. Работает на 1,024 точках на облако.

### Шаг 2: позиционное кодирование

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

Умножение на `2^l * pi` дает постепенно более высокие частоты.

### Шаг 3: небольшой NeRF MLP

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

Крошечный по сравнению с исходным NeRF (у которого 2 MLP-ствола глубины 8). Достаточно, чтобы продемонстрировать архитектуру.

### Шаг 4: объемный рендеринг вдоль луча

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) distances along the ray
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

Один луч, 64 отсчета, композиция в один RGB-пиксель и глубину.

## Используйте это

Для реальной работы:

- `nerfstudio` (Tancik et al.) — текущая референсная библиотека для NeRF / Instant-NGP / Gaussian Splatting. Командная строка плюс веб-просмотрщик.
- `pytorch3d` (Meta) — дифференцируемый рендеринг, утилиты для облаков точек, операции с mesh.
- `open3d` — обработка облаков точек, регистрация, визуализация.

Для развертывания 3D Gaussian splatting в значительной степени заменил чистые NeRF, потому что рендерится в 100x быстрее. Качество реконструкции сопоставимо.

## Доведите до результата

Этот урок создает:

- `outputs/prompt-3d-task-router.md` — промпт, который направляет к правильному 3D-представлению (облако точек, mesh, voxel, NeRF, Gaussian splat) на основе задачи и входных данных.
- `outputs/skill-point-cloud-loader.md` — навык, который пишет PyTorch `Dataset` для файлов .ply / .pcd / .xyz с корректной нормализацией, центрированием и выборкой точек.

## Упражнения

1. **(Легко)** Покажите, что PointNet инвариантен к перестановкам: пропустите одно и то же облако дважды, один раз с перемешанными точками. Проверьте, что выходы идентичны с точностью до шума чисел с плавающей точкой.
2. **(Средне)** Реализуйте минимальную функцию генерации лучей, которая по внутренним параметрам камеры и позе создает начала и направления лучей для каждого пикселя изображения H x W.
3. **(Сложно)** Обучите TinyNeRF на синтетическом датасете отрендеренных видов цветного куба (созданных через дифференцируемый рендеринг или простой трассировщик лучей). Сообщите loss рендеринга на эпохах 1, 10 и 100. На какой эпохе модель начинает создавать узнаваемые виды?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Point cloud | "3D points from LIDAR" | Неупорядоченное множество (x, y, z) + опциональные признаки для каждой точки |
| PointNet | "First neural net on point clouds" | Общий MLP на точку + симметричный (max) pool; инвариантен к перестановкам по конструкции |
| NeRF | "MLP that is the scene" | Сеть, отображающая (x, y, z, dir) в (density, colour); рендерится трассировкой лучей |
| Positional encoding | "Fourier features" | Кодирует каждую координату в sin/cos на нескольких частотах, чтобы преодолеть низкочастотное смещение MLP |
| Volumetric rendering | "Ray integration" | Композиция отсчетов вдоль луча в один пиксель с использованием пропускания и alpha |
| Instant-NGP | "Hash-grid NeRF" | Заменяет координатный MLP NeRF многомасштабной hash grid; в 100-1000x быстрее |
| 3D Gaussian splatting | "Millions of Gaussians" | Сцена = набор 3D-гауссиан; рендерится в реальном времени, обучается за минуты |
| SDF | "Signed distance field" | Функция, возвращающая signed distance до ближайшей поверхности; еще одно неявное представление |

## Дополнительное чтение

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) — классификатор, инвариантный к перестановкам
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) — статья, которая превратила 3D-реконструкцию по фотографиям в задачу для нейронной сети
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989) — hash grids, ускорение в 1000x
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) — архитектура, которая заменила NeRF в production
