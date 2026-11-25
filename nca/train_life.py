"""Ön eğitim: şekilsiz "en basit canlı" (docs/superpowers/specs/2026-11-25-yasayan-ekosistem-design.md, 5).

Hedef görüntü yoktur. Tek bir canlı tek hücreden, düzgün kaynaklı küçük bir dünyada büyür
(yeme yok, g = 0). Kayıp yalnız canlılık ölçütleridir:
  mass    toplam kütle [A_min, A_max] içinde
  debris  min_viable'dan küçük bileşenlerdeki kütle (tek beden zorlanmaz: bölünmek cezasızdır)
  energy  kütle başına enerji e_min'in üstünde
  homeo   beden çerçevesinde durağanlık: ağırlık merkezine göre betimleyiciler son homeo_k adımda
          değişmez (hareket cezalandırılmaz, ödüllendirilmez de)
  bounds  RGB [0, 1], gizli |h| <= 3
Sample pool, hasar (daire / yarım düzlem) ve parametre başına gradyan normalizasyonu
train_ancestor.py ile aynıdır.

Kullanım: python -m nca.train_life configs/life.yaml --seed 0 [--set k=v ...]
"""

import argparse
import csv
import math
import time
from dataclasses import asdict

import numpy as np
import torch
import yaml
from scipy import ndimage

from .evo import make_seed
from .life import E_CH, H0, M_CH, RGB, LifeCA, LifePhysics, life_step
from .model import ALIVE_THRESHOLD
from .train import normalize_grads
from .train_evo import apply_damage
from .utils import ROOT, device, load_config, save_grid, set_seed, to_uint8


def debris_loss(x, min_viable):
    """min_viable'dan küçük bağlı bileşenlerdeki kütle / toplam kütle. (B,)"""
    m = x[:, M_CH]
    alive = (m > ALIVE_THRESHOLD).detach().cpu().numpy()
    small = np.zeros(alive.shape, bool)
    for i, a in enumerate(alive):
        lab, n = ndimage.label(a, structure=np.ones((3, 3)))
        if n:
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            small[i] = np.isin(lab, 1 + np.nonzero(sizes < min_viable)[0])
    small = torch.from_numpy(small).to(x.device).to(x.dtype)
    return (m * small).flatten(1).sum(1) / (m.flatten(1).sum(1) + 1e-6)


def body_descriptors(x):
    """Ağırlık merkezine göre beden betimleyicileri (B, 6): log(1 + kütle), kütle ağırlıklı RGB (3),
    uzama ((λ1 - λ2) / (λ1 + λ2), ikinci momentlerden), kütle başına enerji. Ötelemeyle değişmez."""
    m = x[:, M_CH].clamp(min=0)
    b, h, w = m.shape
    M = m.flatten(1).sum(1) + 1e-6
    yy = torch.arange(h, device=x.device, dtype=x.dtype)[None, :, None]
    xx = torch.arange(w, device=x.device, dtype=x.dtype)[None, None, :]
    cy = (m * yy).flatten(1).sum(1) / M
    cx = (m * xx).flatten(1).sum(1) / M
    dy, dx = yy - cy[:, None, None], xx - cx[:, None, None]
    a = (m * dy * dy).flatten(1).sum(1) / M
    c = (m * dx * dx).flatten(1).sum(1) / M
    bb = (m * dy * dx).flatten(1).sum(1) / M
    diff = torch.sqrt(((a - c) / 2) ** 2 + bb**2 + 1e-12)
    ecc = 2 * diff / (a + c + 1e-6)
    rgb = (x[:, RGB] * m[:, None]).flatten(2).sum(2) / M[:, None]
    e = x[:, E_CH].flatten(1).sum(1) / M
    return torch.cat([torch.log1p(M)[:, None], rgb, ecc[:, None], e[:, None]], 1)


def life_losses(desc_prev, x, cfg):
    """Canlılık kayıpları; her biri (B,). 'total' ağırlıklı toplamdır (cfg['weights'], varsayılan 1)."""
    m = x[:, M_CH]
    M = m.flatten(1).sum(1)
    lo, hi = cfg["A_min"], cfg["A_max"]
    rgb, hid = x[:, RGB], x[:, H0:]
    out = {
        "mass": ((torch.relu(lo - M) + torch.relu(M - hi)) / lo) ** 2,
        "debris": debris_loss(x, cfg["min_viable"]),
        "energy": torch.relu(cfg["e_min"] - x[:, E_CH].flatten(1).sum(1) / (M + 1e-6)) ** 2,
        "homeo": ((body_descriptors(x) - desc_prev) ** 2).sum(1),
        "bounds": 10 * (torch.relu(-rgb) ** 2 + torch.relu(rgb - 1) ** 2).flatten(1).mean(1)
        + (torch.relu(hid.abs() - 3) ** 2).flatten(1).mean(1),
    }
    wts = cfg.get("weights", {})
    out["total"] = sum(wts.get(k, 1.0) * v for k, v in out.items())
    return out


@torch.no_grad()
def _torus_shift(w, n):
    """Kütle ağırlıklı dairesel ortalama konumu (n uzunluklu eksen) -> merkeze taşıyan tam sayı kayma."""
    a = torch.arange(n, device=w.device, dtype=w.dtype) * (2 * math.pi / n)
    c = torch.atan2((w * torch.sin(a)).sum(1), (w * torch.cos(a)).sum(1)) * n / (2 * math.pi)
    return torch.round(n // 2 - c).long() % n


def center_torus(x, R):
    """Sarmal arenada her örneği kütle merkezi ortaya gelecek biçimde kaydırır (fizik kaymayla değişmez);
    kenardan geçen beden kayıplarda ikiye bölünmüş sayılmasın. (x, R) döndürür."""
    m = x[:, M_CH].clamp(min=0)
    sy, sx = _torus_shift(m.sum(2), m.shape[1]), _torus_shift(m.sum(1), m.shape[2])
    xs = [torch.roll(x[i], (int(sy[i]), int(sx[i])), (1, 2)) for i in range(len(x))]
    Rs = [torch.roll(R[i], (int(sy[i]), int(sx[i])), (1, 2)) for i in range(len(R))]
    return torch.stack(xs), torch.stack(Rs)


def run_life(model, x, g, R, phys, steps, segment):
    """Gradyan checkpoint'li açılım. (x, g, R) döndürür."""
    from torch.utils.checkpoint import checkpoint

    def seg(x, g, R, k):
        for _ in range(k):
            x, g, R, _, _ = life_step(model, x, g, R, phys)
        return x, g, R

    done = 0
    while done < steps:
        k = min(segment, steps - done) if segment > 0 else steps - done
        x, g, R = checkpoint(seg, x, g, R, k, use_reentrant=False) if segment > 0 else seg(x, g, R, k)
        done += k
    return x, g, R


def seed_state(model, b, size, dev):
    """Ortada tek hücre (m = 1, E = 1, gizli = 1), g = 0, R = 1."""
    x, g = make_seed(torch.zeros(b, model.n_genome, device=dev), size, model.channels)
    return x, g, torch.ones(b, 1, size, size, device=dev)


def render(x):
    """(B, C, H, W) -> premultiplied RGBA (B, 4, H, W): alfa = clamp(m, 0, 1)."""
    a = x[:, M_CH : M_CH + 1].clamp(0, 1)
    return torch.cat([x[:, RGB].clamp(0, 1) * a, a], 1)


def load_life(run_dir, map_location=None):
    """Eğitilmiş koşudan (model, phys) yükler."""
    ck = torch.load(ROOT / run_dir / "model.pt", map_location=map_location or device(), weights_only=False)
    model = LifeCA(**ck["model_config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(map_location or device()).eval(), LifePhysics(**ck["physics"])


def train(cfg):
    dev = device()
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 91)
    out = ROOT / cfg.get("out_dir", "runs") / f"{cfg['name']}_s{cfg['seed']}"
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    size, bs = cfg["grid"], cfg["batch_size"]
    model = LifeCA(**cfg["model"], film_scale=0.0).to(dev)  # ön eğitimde g = 0: FiLM etkisiz
    phys = LifePhysics(**cfg.get("physics", {}))
    if cfg.get("init_from"):
        model.load_state_dict(torch.load(ROOT / cfg["init_from"], map_location=dev, weights_only=False)["state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [cfg["lr_drop_iter"]], gamma=cfg["lr_drop"])

    P = cfg["pool_size"]
    pool_x, pool_g, pool_R = seed_state(model, P, size, dev)
    pool_loss = torch.full((P,), float("inf"), device=dev)
    smin, smax = cfg["steps"]
    hk, dcfg = cfg["homeo_k"], cfg["damage"]

    log_f = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    names = ["mass", "debris", "energy", "homeo", "bounds"]
    log.writerow(["iter", "loss", *names, "mean_mass", "sec"])
    t0 = time.time()
    print(f"[{out.name}] parametre={model.n_params()}", flush=True)
    for it in range(cfg["iterations"]):
        idx = torch.randint(P, (bs,), generator=gen).to(dev)
        idx = idx[pool_loss[idx].argsort(descending=True)]
        x, g, R = pool_x[idx].clone(), pool_g[idx].clone(), pool_R[idx].clone()
        x[:1], g[:1], R[:1] = seed_state(model, 1, size, dev)  # en kötü örnek: yeni tohum
        if phys.torus:  # bedenler ortada: kenardan geçmez, hasar ve kayıplar tek beden görür
            x, R = center_torus(x, R)
        if it >= dcfg["start_iter"]:
            n = dcfg["n_damaged"]
            apply_damage(x[bs - n :], g[bs - n :], gen, dcfg, phys)
        steps = int(torch.randint(smin, smax + 1, (1,), generator=gen))
        seg = cfg.get("checkpoint_segment", 16)
        # g = None: ön eğitimde genom sıfır ve FiLM kapalı, genom taşınmaz (pool'daki g sıfır kalır)
        x, _, R = run_life(model, x, None, R, phys, steps - hk, seg)
        desc_prev = body_descriptors(x)
        x, _, R = run_life(model, x, None, R, phys, hk, seg)
        L = life_losses(desc_prev, x, cfg)
        loss = L["total"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        normalize_grads(model)
        opt.step()
        sched.step()
        pool_x[idx], pool_R[idx] = x.detach(), R.detach()
        pool_loss[idx] = L["total"].detach()
        mean_mass = x[:, M_CH].flatten(1).sum(1).mean().item()
        if it % cfg.get("log_every", 50) == 0 or it == cfg["iterations"] - 1:
            log.writerow([it, f"{loss.item():.6f}", *[f"{L[k].mean().item():.6f}" for k in names],
                          f"{mean_mass:.1f}", f"{time.time() - t0:.1f}"])
            log_f.flush()
        if it % cfg.get("print_every", 500) == 0:
            parts = " ".join(f"{k}={L[k].mean().item():.4f}" for k in names)
            print(f"[{out.name}] it={it} loss={loss.item():.4f} {parts} kütle={mean_mass:.0f} t={time.time() - t0:.0f}s", flush=True)
        if it % cfg.get("snapshot_every", 1000) == 0 or it == cfg["iterations"] - 1:
            with torch.no_grad():
                xe, ge, Re = seed_state(model, 8, size, dev)
                frames = []
                for s in range(400):
                    xe, _, Re, _, _ = life_step(model, xe, None, Re, phys)
                    if s in (49, 399):
                        frames += list(to_uint8(render(xe)))
            save_grid(out / "snapshots" / f"gallery_{it:05d}.png", frames, scale=2, ncol=8)
    log_f.close()
    torch.save({"state_dict": model.state_dict(), "model_config": model.config(), "physics": asdict(phys), "config": cfg},
               out / "model.pt")
    print(f"[{out.name}] bitti {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
