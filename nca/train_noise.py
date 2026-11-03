"""Genişletme 2 (esnek): rastgele gürültüden birden fazla bireyin oluşması.

Başlangıç durumu tek tohum yerine seyrek rastgele gürültüdür. T adım sonra alfa kanalında
bağlı bileşenler bulunur; en büyük K bileşenin her biri için ağırlık merkezine hizalanmış
40x40 pencere hedefle karşılaştırılır. Bileşen ataması türevlenebilir değildir ama atama
sabitlendikten sonra her pencerenin MSE'si türevlenebilir.

Kayıp = pencere başına MSE ortalaması + bg_weight * pencere dışı RGBA^2 ortalaması.
Bileşen sayısı K'dan azsa eksik bireyler için boş bir yere "sanal" pencere konur; bu, orada
birey büyütmeye doğru gradyan verir (birey sayısı sapma cezasının türevlenebilir vekili).

Kullanım: python -m nca.train_noise configs/noise.yaml --seed 0
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.measure import label, regionprops

from .model import CAModel
from .train import normalize_grads, run_steps
from .utils import ROOT, device, load_config, load_target, save_grid, set_seed, to_uint8


def noise_state(n, size, channels, density, dev, gen):
    """Seyrek gürültü: hücrelerin `density` kadarı rastgele durumda, kalanı boş."""
    on = (torch.rand(n, 1, size, size, generator=gen) < density).float()
    x = torch.rand(n, channels, size, size, generator=gen) * on
    return x.to(dev)


def windows_for(alpha, k, th, tw, rng, min_area=20):
    """Bir örnek için en fazla k pencerenin sol üst köşeleri: gerçek bileşenler + sanal yerler."""
    h, w = alpha.shape
    comps = sorted(
        (r for r in regionprops(label(alpha > 0.1, connectivity=2)) if r.area >= min_area),
        key=lambda r: -r.area,
    )[:k]
    boxes = []
    for r in comps:
        cy, cx = r.centroid
        y0 = int(np.clip(round(cy - th / 2), 0, h - th))
        x0 = int(np.clip(round(cx - tw / 2), 0, w - tw))
        boxes.append((y0, x0))
    tries = 0
    while len(boxes) < k and tries < 50:
        tries += 1
        y0, x0 = int(rng.integers(0, h - th + 1)), int(rng.integers(0, w - tw + 1))
        if all(abs(y0 - a) >= th or abs(x0 - b) >= tw for a, b in boxes):
            boxes.append((y0, x0))
    return boxes, len(comps)


def noise_loss(x, target, k, bg_weight, rng):
    th, tw = target.shape[1:]
    losses, counts = [], []
    for i in range(x.shape[0]):
        boxes, n = windows_for(x[i, 3].detach().cpu().numpy(), k, th, tw, rng)
        counts.append(n)
        inside = torch.zeros_like(x[i, 3], dtype=torch.bool)
        win = []
        for y0, x0 in boxes:
            win.append(((x[i, :4, y0 : y0 + th, x0 : x0 + tw] - target) ** 2).mean())
            inside[y0 : y0 + th, x0 : x0 + tw] = True
        bg = (x[i, :4, ~inside] ** 2).mean() if (~inside).any() else x.sum() * 0
        losses.append(torch.stack(win).mean() + bg_weight * bg)
    return torch.stack(losses).mean(), counts


def train(cfg, out_dir=None):
    dev = device()
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 999)
    rng = np.random.default_rng(cfg["seed"])
    out_dir = Path(out_dir or ROOT / cfg.get("out_dir", "runs") / f"{cfg['name']}_s{cfg['seed']}")
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    target = load_target(cfg.get("target"))["rgba"].to(dev)
    m = cfg["model"]
    model = CAModel(m["n_hidden"], m["hidden_dim"], m["perception"], m["fire_rate"]).to(dev)
    if cfg.get("init_from"):
        # büyümeyi bilen bir modelden ince ayar: sıfırdan eğitimde model her şeyi öldürüp
        # gradyansız ölü duruma düşebiliyor
        model.load_state_dict(torch.load(ROOT / cfg["init_from"], map_location=dev, weights_only=False)["state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [cfg["lr_drop_iter"]], gamma=cfg["lr_drop"])
    nz = cfg["noise"]
    size, bs, k = cfg["grid"], cfg["batch_size"], nz["n_individuals"]

    def fresh(n):
        return noise_state(n, size, model.channels, nz["density"], dev, gen)

    pool = fresh(cfg["pool"]["size"])
    smin, smax = cfg["steps"]
    log_f = open(out_dir / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "mean_count", "sec"])
    t0 = time.time()
    for it in range(cfg["iterations"]):
        idx = torch.randint(len(pool), (bs,), generator=gen).to(dev)
        x = pool[idx].clone()
        x[: nz.get("n_fresh", 2)] = fresh(nz.get("n_fresh", 2))
        x = run_steps(model, x, int(torch.randint(smin, smax + 1, (1,), generator=gen)), cfg.get("checkpoint_segment", 0))
        loss, counts = noise_loss(x, target, k, nz["bg_weight"], rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        normalize_grads(model)
        opt.step()
        sched.step()
        x = x.detach()
        dead = (x[:, 3] > 0.1).flatten(1).sum(1) == 0
        if dead.any():  # ölü örnekler havuzda kalırsa gradyan vermez; taze gürültüyle değiştir
            x[dead] = fresh(int(dead.sum()))
        pool[idx] = x
        if it % cfg.get("log_every", 50) == 0:
            log.writerow([it, f"{loss.item():.6f}", np.mean(counts), f"{time.time() - t0:.1f}"])
            log_f.flush()
        if it % cfg.get("snapshot_every", 1000) == 0 or it == cfg["iterations"] - 1:
            save_grid(out_dir / "snapshots" / f"batch_{it:05d}.png", to_uint8(x), scale=2)
    log_f.close()
    torch.save({"state_dict": model.state_dict(), "model_config": model.config(), "config": cfg}, out_dir / "model.pt")
    print(f"[{out_dir.name}] bitti, {time.time() - t0:.0f}s")
    return out_dir


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
