"""E1/E4 - Çeşitlilik, canlılık ve parçadan yenilenme (evrimsel fizik).

E1: N rastgele genomdan tek hücreyle büyüyen canlılar: canlılık (alan aralığı, tek beden,
1000 adım kararlılığı), çeşitlilik (morfolojik tanımlayıcıların yayılımı, fenotipler arası
hizalı IoU), genomun fenotipten okunabilirliği (çözücü R^2).
E4: aynı canlı kesilir; mutasyonsuz ve mutasyonlu (mu) yenilenme sonrası canlılık ve
ebeveyne fenotip mesafesi; mesafenin yara potansiyeliyle ilişkisi.

Çıktı: experiments/results/e1.csv, e4.csv, experiments/results/e1/<koşu>_gallery.png
"""

import argparse
import csv
import math

import numpy as np
import torch

from nca import damage as dmg
from nca.evo import V_CH, EvoCA, Physics, alive_cells, make_seed, mutate, step, wound
from nca.metrics import aligned_iou, count_individuals
from nca.train_evo import Decoder
from nca.utils import ROOT, device, save_grid, set_seed, to_uint8

RES = ROOT / "experiments" / "results"


def load_evo(run_dir, dev):
    ck = torch.load(run_dir / "model.pt", map_location=dev, weights_only=False)
    model = EvoCA(**ck["model_config"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    phys = Physics(**ck["physics"])
    dec = None
    if "decoder" in ck:  # hedefsiz eğitimde var; ata eğitiminde yok
        dec = Decoder(4, model.n_genome).to(dev).eval()
        dec.load_state_dict(ck["decoder"])
    return model, phys, dec, ck["config"]


@torch.no_grad()
def grow(model, phys, genomes, size, steps, mu=0.0, snaps=()):
    x, g = make_seed(genomes, size, model.channels)
    out = {}
    for t in range(1, steps + 1):
        x, g = step(model, x, g, phys)
        if mu:
            g = mutate(g, x, mu)
        if t in snaps:
            out[t] = (x.clone(), g.clone())
    return x, g, out


def descriptors(x):
    """Morfolojik tanımlayıcılar: alan, uzama, doluluk, ortalama renk, renk std."""
    rows = []
    for s in x:
        a = s[3].clamp(0, 1).cpu().numpy()
        m = a > 0.1
        if m.sum() < 3:
            rows.append([0, 0, 0, 0, 0, 0, 0])
            continue
        yx = np.argwhere(m).astype(float)
        ev = np.sort(np.linalg.eigvalsh(np.cov(yx.T) + 1e-6 * np.eye(2)))
        y0, x0 = yx.min(0)
        y1, x1 = yx.max(0)
        fill = m.sum() / ((y1 - y0 + 1) * (x1 - x0 + 1))
        rgb = (s[:3] / s[3:4].clamp(min=0.5)).cpu().numpy()[:, m]
        rows.append([m.sum(), math.sqrt(ev[1] / ev[0]), fill, *rgb.mean(1), rgb.std(1).mean()])
    return np.array(rows)


@torch.no_grad()
def e1(run_dir, n, size, out):
    dev = device()
    model, phys, dec, cfg = load_evo(run_dir, dev)
    amin, amax = cfg.get("area", (100, 900))
    genomes = torch.randn(n, model.n_genome, generator=torch.Generator().manual_seed(2026)).to(dev)
    x, g, snaps = grow(model, phys, genomes, size, 1000, snaps=(200,))
    x200 = snaps[200][0]
    save_grid(out / f"{run_dir.name}_gallery.png", to_uint8(x200[:64, :4]), scale=3, ncol=8)
    area = (x200[:, 3] > 0.1).flatten(1).sum(1).cpu().numpy()
    single = np.array([count_individuals((s[3] > 0.1).cpu().numpy()) == 1 for s in x200])
    in_range = (area >= amin * 0.8) & (area <= amax * 1.2)
    masks = [(s[3] > 0.1).cpu().numpy() for s in x200]
    m1000 = [(s[3] > 0.1).cpu().numpy() for s in x]
    stable = np.array([aligned_iou(a, b) for a, b in zip(masks, m1000)])
    k = min(n, 40)
    pair = [aligned_iou(masks[i], masks[j]) for i in range(k) for j in range(i + 1, k)]
    if dec is not None:
        pred = dec(x200[:, :4].clamp(0, 1))
        r2 = (1 - ((pred - genomes) ** 2).sum(0) / ((genomes - genomes.mean(0)) ** 2).sum(0)).mean().item()
    else:
        r2 = float("nan")
    d = descriptors(x200)
    viable = in_range & single & (stable > 0.8)
    extra = {}
    if "n_bank" in cfg:  # ata modeli: her genom kendi ata hedefiyle karşılaştırılır (eğitimde görülmemiş genomlar)
        import sys

        sys.path.insert(0, str(ROOT / "targets"))
        import creature_space as cs

        pad = (size - cs.SIZE) // 2
        tg = np.stack([np.pad(cs.render(gv), ((pad, size - cs.SIZE - pad), (pad, size - cs.SIZE - pad), (0, 0))) for gv in genomes.cpu().numpy()])
        tmask = tg[..., 3] > 0.1
        ious = np.array([np.logical_and(m, t).sum() / max(1, np.logical_or(m, t).sum()) for m, t in zip(masks, tmask)])
        mse = ((x200[:, :4].permute(0, 2, 3, 1).cpu().numpy() - tg) ** 2).mean((1, 2, 3))
        extra = {"target_iou_mean": float(ious.mean()), "target_iou_ge_0.9": float((ious >= 0.9).mean()), "target_mse": float(mse.mean())}
        save_grid(out / f"{run_dir.name}_vs_target.png",
                  [f for pair in zip(to_uint8(torch.from_numpy(tg[:16]).permute(0, 3, 1, 2)), to_uint8(x200[:16, :4])) for f in pair], scale=3, ncol=8)
    return {
        "run": run_dir.name,
        "viable": float(viable.mean()),
        "area_in_range": float(in_range.mean()),
        "single_body": float(single.mean()),
        "stable_iou_200_1000": float(stable.mean()),
        "pairwise_shape_iou_mean": float(np.mean(pair)),
        "pairwise_shape_iou_min": float(np.min(pair)),
        "decoder_r2": r2,
        "elongation_std": float(d[:, 1].std()),
        "fill_std": float(d[:, 2].std()),
        "hue_spread": float(d[:, 3:6].std(0).mean()),
        **extra,
    }, genomes, x200


@torch.no_grad()
def e4(run_dir, genomes, grown, size, out, mus=(0.0, 0.15, 0.3), keep=0.5, steps=300):
    dev = device()
    model, phys, _, _ = load_evo(run_dir, dev)
    rows = []
    gen = torch.Generator().manual_seed(7)
    for mu in mus:
        x, g = grown.clone(), make_seed(genomes, size, model.channels)[1]
        # büyümüş durumun genom alanını yeniden kur: canlı hücrelere organizma genomu
        g = genomes[:, :, None, None] * alive_cells(x).float()
        before = alive_cells(x)
        for i in range(len(x)):
            dmg.halfplane_cut(x[i], float(torch.rand(1, generator=gen)) * 2 * math.pi, keep)
        g = g * alive_cells(x).float()
        x = wound(x, before & ~alive_cells(x), phys)
        vmax = []
        for t in range(steps):
            x, g = step(model, x, g, phys)
            g = mutate(g, x, mu)
            vmax.append(x[:, V_CH].amax((1, 2)))
        masks0 = [(s[3] > 0.1).cpu().numpy() for s in grown]
        masks1 = [(s[3] > 0.1).cpu().numpy() for s in x]
        iou = np.array([aligned_iou(a, b) for a, b in zip(masks0, masks1)])
        single = np.array([count_individuals(m) == 1 for m in masks1])
        gdrift = ((g.sum((2, 3)) / alive_cells(x).float().sum((2, 3)).clamp(min=1)) - genomes).norm(dim=1).cpu().numpy()
        colour = (x[:, :3].mean((2, 3)) - grown[:, :3].mean((2, 3))).norm(dim=1).cpu().numpy()
        rows.append(
            {
                "run": run_dir.name,
                "mu": mu,
                "regrown_single": float(single.mean()),
                "shape_iou_to_parent": float(iou.mean()),
                "genome_drift": float(gdrift.mean()),
                "colour_change": float(colour.mean()),
                "V_peak": float(torch.stack(vmax).amax(0).mean()),
            }
        )
        if mu == mus[-1]:
            save_grid(out / f"{run_dir.name}_regen_mu{mu}.png",
                      [f for pair in zip(to_uint8(grown[:8, :4]), to_uint8(x[:8, :4])) for f in pair], scale=3, ncol=8)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["ancestor_s*", "evo_s*"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--grid", type=int, default=64)
    args = ap.parse_args()
    set_seed(0)
    out = RES / "e1"
    out.mkdir(parents=True, exist_ok=True)
    runs = [d for pat in args.runs for d in sorted((ROOT / "runs").glob(pat)) if (d / "model.pt").exists()]
    r1, r4 = [], []
    for r in runs:
        row, genomes, grown = e1(r, args.n, args.grid, out)
        print(row)
        r1.append(row)
        for row4 in e4(r, genomes, grown, args.grid, out):
            print(row4)
            r4.append(row4)
    for name, rows in (("e1.csv", r1), ("e4.csv", r4)):
        with open(RES / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"yazıldı: experiments/results/{name}")


if __name__ == "__main__":
    main()
