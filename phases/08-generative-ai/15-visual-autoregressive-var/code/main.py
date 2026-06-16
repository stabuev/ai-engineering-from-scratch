import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

GRID = 4
SCALES = [1, 2, 4]
CODEBOOK = 8
N_DATA = 512
D_MODEL = 32
HEADS = 2
LAYERS = 2
EPOCHS = 300


def make_rings(n):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing="ij")
    imgs = []
    for _ in range(n):
        cy, cx = torch.rand(2) * 1.2 - 0.6
        radius = 0.3 + torch.rand(1) * 0.6
        dist = ((ys - cy) ** 2 + (xs - cx) ** 2).sqrt()
        imgs.append(torch.exp(-((dist - radius) ** 2) / 0.05))
    return torch.stack(imgs)


def pool_to(img, s):
    return F.adaptive_avg_pool2d(img.unsqueeze(1), s).squeeze(1)


def upsample_to(grid, s):
    return F.interpolate(grid.unsqueeze(1), size=(s, s), mode="nearest").squeeze(1)


def build_codebook(data):
    residual = data.clone()
    values = []
    for s in SCALES:
        pooled = pool_to(residual, s)
        values.append(pooled.reshape(-1))
        residual = residual - upsample_to(pooled, GRID)
    allv = torch.cat(values)
    qs = torch.linspace(0.02, 0.98, CODEBOOK)
    return torch.quantile(allv, qs)


def tokenize(img, codebook):
    residual = img.clone()
    grids = []
    recon = torch.zeros_like(img)
    for s in SCALES:
        pooled = pool_to(residual, s)
        idx = (pooled.unsqueeze(-1) - codebook).abs().argmin(-1)
        quant = codebook[idx]
        grids.append(idx)
        recon = recon + upsample_to(quant, GRID)
        residual = residual - upsample_to(quant, GRID)
    return grids, recon


def scale_of_positions():
    scale_id, rows, cols = [], [], []
    for si, s in enumerate(SCALES):
        for r in range(s):
            for c in range(s):
                scale_id.append(si)
                rows.append(r)
                cols.append(c)
    return torch.tensor(scale_id), torch.tensor(rows), torch.tensor(cols)


def scale_ordered_mask(scale_id):
    a = scale_id.unsqueeze(0)
    b = scale_id.unsqueeze(1)
    return b >= a


class VAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(CODEBOOK + 1, D_MODEL)
        self.scale_emb = nn.Embedding(len(SCALES), D_MODEL)
        self.pos_emb = nn.Embedding(GRID * GRID, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, D_MODEL * 2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Linear(D_MODEL, CODEBOOK)
        sid, rows, cols = scale_of_positions()
        self.register_buffer("sid", sid)
        self.register_buffer("posid", rows * GRID + cols)
        self.register_buffer("attn_mask", ~scale_ordered_mask(sid))

    def inputs_from_lower(self, grids):
        batch = grids[0].shape[0]
        feats = []
        recon_emb = torch.zeros(batch, GRID, GRID, D_MODEL)
        for si, s in enumerate(SCALES):
            if si == 0:
                start = self.tok(torch.full((batch, 1), CODEBOOK))
                feats.append(start)
            else:
                pooled = F.adaptive_avg_pool2d(
                    recon_emb.permute(0, 3, 1, 2), s
                ).permute(0, 2, 3, 1).reshape(batch, s * s, D_MODEL)
                feats.append(pooled)
            emb = self.tok(grids[si].reshape(batch, s * s)).reshape(batch, s, s, D_MODEL)
            recon_emb = recon_emb + F.interpolate(
                emb.permute(0, 3, 1, 2), size=(GRID, GRID), mode="nearest"
            ).permute(0, 2, 3, 1)
        return torch.cat(feats, dim=1)

    def forward(self, x):
        x = x + self.scale_emb(self.sid) + self.pos_emb(self.posid)
        h = self.enc(x, mask=self.attn_mask)
        return self.head(h)


def flatten_targets(grids):
    return torch.cat([g.reshape(g.shape[0], -1) for g in grids], dim=1)


def train():
    data = make_rings(N_DATA)
    codebook = build_codebook(data)
    grids, recon = tokenize(data, codebook)
    rec_err = (data - recon).pow(2).mean().item()
    print(f"Multi-scale VQ tokenizer: scales {SCALES}, codebook {CODEBOOK}")
    print(f"  reconstruction MSE: {rec_err:.4f}")

    model = VAR()
    mask = model.attn_mask
    print(f"\nScale-ordered attention mask ({mask.shape[0]} positions, True = blocked):")
    print(f"  scale-1 (1 tok) attends only to scale 1; scale-3 (16 tok) attends to all scales")
    print(f"  same-scale attention is open -> parallel within scale")

    targets = flatten_targets(grids)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(EPOCHS):
        x = model.inputs_from_lower(grids)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, CODEBOOK), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 60 == 0 or epoch == EPOCHS - 1:
            acc = (logits.argmax(-1) == targets).float().mean().item()
            print(f"  epoch {epoch:4d} | loss {loss.item():.4f} | token acc {acc:.3f}")

    print("\nGeneration: one transformer pass per scale, parallel within scale")
    with torch.no_grad():
        x = model.inputs_from_lower([g[:1] for g in grids])
        logits = model(x)
    offs, passes = 0, []
    for si, s in enumerate(SCALES):
        pred = logits[:1, offs:offs + s * s].argmax(-1).reshape(s, s)
        passes.append(pred)
        offs += s * s
        print(f"  pass {si + 1}: predicted scale {s}x{s} ({s * s} tokens in one forward)")
    print(f"  total passes: {len(SCALES)}  vs per-token AR would need {sum(s * s for s in SCALES)}")


if __name__ == "__main__":
    train()
