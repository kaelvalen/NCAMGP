"""E5 - Tek soyluk uygunluk testi ve kalıtsal uygunluk farkı.

Rekabet testi (E3) yerleşim piyangosu yüzünden çok gürültülü. Burada her genom tek başına
taze, standart bir dünyaya tek tohum olarak ekilir (mutasyon kapalı); T adım sonra soyun birey ve
hücre sayısı ölçülür. Aynı genom farklı rastgelelik tohumlarıyla tekrarlanır.

  variance  N rastgele ata genomu x R tekrar: genomlar arası / toplam varyans (geniş anlamda
            kalıtsallık H^2, tek yönlü ANOVA). H^2 ~ 0 ise seçilimin ayıklayabileceği fark yok.
  confirm   verify adaylarının en iyileri, çok tekrarla.
  verify    eğrinin en iyi son genomları ile tüm kurucular, taze tekrarlarla.
  curve     E3'ün kaydettiği popülasyon örnekleri (experiments/results/e3/*_snapshots.npz):
            her zaman noktasındaki genomların uygunluğu. Evrim işliyorsa uygunluk zamanla artar;
            mutasyonsuz kontrolde yalnız başlangıç çeşitliliği içinden seçilim kadar artar.

Çıktı: e5_variance.csv, e5_summary.csv (variance); e5_curve.csv (curve)
"""

import argparse
import csv
import time
from dataclasses import replace

import numpy as np
import torch

from nca.utils import ROOT, device
from nca.world import World, WorldParams

from e1_diversity import load_evo  # noqa: E402

RES = ROOT / "experiments" / "results"


@torch.no_grad()
def assay(model, phys, genome, seed, steps, size, params):
    """Tek tohumluk soy: T adım sonra birey sayısı, hücre sayısı, toplam doğum."""
    g = torch.as_tensor(np.asarray(genome, np.float32), device=device())[None]
    w = World(model, phys, size=size, params=replace(params, mu=0.0), seed=seed, genomes=g)
    for _ in range(steps):
        w.step()
    h = w.history[-1]
    return {"n_org": h["n_org"], "cells": h["cells"], "births": sum(1 for e in w.events if e[1] == "dogum"),
            "mean_area": h["mean_area"]}


def h2(values, groups):
    """Tek yönlü ANOVA ile geniş anlamda kalıtsallık: var_arası / (var_arası + var_içi)."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    ids = np.unique(groups)
    r = np.mean([np.sum(groups == i) for i in ids])
    means = np.array([values[groups == i].mean() for i in ids])
    within = np.mean([values[groups == i].var(ddof=1) for i in ids])
    between = max(0.0, means.var(ddof=1) - within / r)
    return between / (between + within) if between + within > 0 else 0.0


def curve(model, phys, args, params):
    rows = []
    t0 = time.time()
    files = sorted((RES / "e3").glob(f"{args.run}_*_snapshots.npz"))
    if args.only:
        files = [f for f in files if any(f"_{c}_seed" in f.name for c in args.only)]
    for f in files:
        cond = f.name[len(args.run) + 1 :].split("_seed")[0]
        seed = int(f.name.split("_seed")[1].split("_")[0])
        d = np.load(f)
        for key in sorted(d.files, key=lambda k: int(k[1:])):
            gs = d[key][: args.k]
            for gi, gv in enumerate(gs):
                for rep in range(args.reps):
                    res = assay(model, phys, gv, 2000 + rep, args.steps, args.size, params)
                    rows.append({"cond": cond, "seed": seed, "t": int(key[1:]), "genome": gi, "rep": rep, **res})
            sel = [r["n_org"] for r in rows if r["cond"] == cond and r["seed"] == seed and r["t"] == int(key[1:])]
            print(f"{cond} seed={seed} t={key[1:]}: ortalama birey {np.mean(sel):.1f} (n={len(sel)}, {time.time() - t0:.0f}s)", flush=True)
    with open(RES / args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: experiments/results/{args.out}")


def verify(model, phys, args, params):
    """Eğrinin en iyi son genomlarını tüm kuruculara karşı taze tekrarlarla yeniden ölçer.

    Eğrideki maksimumlar 2 tekrarlık gürültülü tahminlerdir (kazananın laneti); burada seçim
    e5_curve*.csv'den yapılır, ölçüm yeni rastgelelik tohumlarıyla (3000+) tekrarlanır.
    """
    import pandas as pd

    c = pd.concat([pd.read_csv(RES / f) for f in ("e5_curve.csv", "e5_curve_mu.csv") if (RES / f).exists()])
    t_end = int(c.t.max())
    rows, t0 = [], time.time()
    for seed in sorted(c.seed.unique()):
        cands = []
        f0 = np.load(RES / "e3" / f"{args.run}_mutasyonsuz_seed{seed}_snapshots.npz")["t0"]
        cands += [("kurucu", gi, gv) for gi, gv in enumerate(f0)]
        for cond in sorted(c.cond.unique()):
            d = c[(c.cond == cond) & (c.seed == seed) & (c.t == t_end)].groupby("genome").n_org.mean()
            snap = np.load(RES / "e3" / f"{args.run}_{cond}_seed{seed}_snapshots.npz")[f"t{t_end}"]
            cands += [(cond, int(gi), snap[int(gi)]) for gi in d.sort_values(ascending=False).index[: args.top]]
        for cond, gi, gv in cands:
            for rep in range(args.reps):
                res = assay(model, phys, gv, 3000 + rep, args.steps, args.size, params)
                rows.append({"cond": cond, "seed": int(seed), "genome": gi, "rep": rep, **res})
            m = np.mean([r["n_org"] for r in rows[-args.reps :]])
            print(f"seed={seed} {cond} genom {gi}: {m:.1f}  ({time.time() - t0:.0f}s)", flush=True)
    with open(RES / "e5_verify.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("yazıldı: experiments/results/e5_verify.csv")


def confirm(model, phys, args, params):
    """verify adayları arasından en iyi evrimleşmiş genomları ve en iyi kurucuları çok tekrarla ölçer.

    verify'daki maksimum da çok adaydan seçildiği için yanlıdır; burada seçim e5_verify.csv'den,
    ölçüm yeni tohumlarla (5000+) ve --reps tekrarla yapılır. Test GPU'da deterministik değildir,
    tekrar sayısı bu gürültüyü de ortalar.
    """
    import pandas as pd

    v = pd.read_csv(RES / "e5_verify.csv")
    m = v.groupby(["cond", "seed", "genome"]).n_org.mean().reset_index()
    evo = m[m.cond.str.startswith("evrim")].sort_values("n_org", ascending=False).head(args.top)
    fou = m[m.cond == "kurucu"].sort_values("n_org", ascending=False).groupby("seed").head(3)
    rows, t0 = [], time.time()
    for _, r in pd.concat([fou, evo]).iterrows():
        cond, seed, gi = r.cond, int(r.seed), int(r.genome)
        if cond == "kurucu":
            gv = np.load(RES / "e3" / f"{args.run}_mutasyonsuz_seed{seed}_snapshots.npz")["t0"][gi]
        else:
            gv = np.load(RES / "e3" / f"{args.run}_{cond}_seed{seed}_snapshots.npz")["t30000"][gi]
        for rep in range(args.reps):
            res = assay(model, phys, gv, 5000 + rep, args.steps, args.size, params)
            rows.append({"cond": cond, "seed": seed, "genome": gi, "rep": rep, **res})
        y = [x["n_org"] for x in rows[-args.reps :]]
        print(f"seed={seed} {cond} genom {gi}: {np.mean(y):.1f} ± {np.std(y, ddof=1) / np.sqrt(len(y)):.1f}  ({time.time() - t0:.0f}s)",
              flush=True)
    with open(RES / "e5_confirm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("yazıldı: experiments/results/e5_confirm.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", choices=["variance", "curve", "verify", "confirm"], default="variance")
    ap.add_argument("--k", type=int, default=8, help="curve: zaman noktası başına genom")
    ap.add_argument("--only", nargs="*", default=None, help="curve: yalnız bu koşullar (ör. evrim_mu0.5)")
    ap.add_argument("--out", default="e5_curve.csv")
    ap.add_argument("--top", type=int, default=4, help="verify: koşul başına en iyi son genom")
    ap.add_argument("--run", default="ancestor_s0")
    ap.add_argument("--n", type=int, default=16, help="genom sayısı")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    model, phys, _, _ = load_evo(ROOT / "runs" / args.run, device())
    params = WorldParams()
    if args.mode == "curve":
        return curve(model, phys, args, params)
    if args.mode == "verify":
        return verify(model, phys, args, params)
    if args.mode == "confirm":
        return confirm(model, phys, args, params)
    genomes = torch.randn(args.n, model.n_genome, generator=torch.Generator().manual_seed(99)).numpy()
    rows = []
    t0 = time.time()
    for gi, gv in enumerate(genomes):
        for rep in range(args.reps):
            res = assay(model, phys, gv, 1000 + rep, args.steps, args.size, params)
            rows.append({"genome": gi, "rep": rep, **res, **{f"g{k}": float(v) for k, v in enumerate(gv)}})
        m = [r["n_org"] for r in rows if r["genome"] == gi]
        print(f"genom {gi}: birey {m}  ({time.time() - t0:.0f}s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "e5_variance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for metric in ("n_org", "cells", "births", "mean_area"):
        vals = [r[metric] for r in rows]
        grp = [r["genome"] for r in rows]
        summ.append({"metric": metric, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "H2": h2(vals, grp)})
        print(summ[-1])
    with open(RES / "e5_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)


if __name__ == "__main__":
    main()
