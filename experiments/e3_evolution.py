"""E3 - Evrimsel iyileşme: ekosistemde mutasyonlu ve mutasyonsuz koşular + ata-torun rekabeti.

1. Koşullar (tohum başına):
     evrim       mu > 0 (yara potansiyeline bağlı mutasyon), yırtıcı kesmeleri açık
     mutasyonsuz mu = 0 (yalnız başlangıç genomları arasında seçilim)
   Popülasyon özeti (birey sayısı, hücre, ortalama yaş, genom çeşitliliği) ve olaylar kaydedilir.
2. Rekabet (torun - kendi atası): evrim koşusunun sonunda hayatta olan en büyük K bireyin
   genomu ile her birinin soyunun kök atasının başlangıç genomu aynı dünyaya eşit sayıda ekilir,
   mutasyon kapalı. Sonda hücrelerin ne kadarı torun grubuna ait (en yakın kurucu genoma göre)
   ölçülür. > 0.5 ise evrimsel iyileşme. Her soydan en fazla bir torun alınır.

Çıktı: experiments/results/e3_history.csv, e3_competition.csv, e3/<koşu>_*.png
"""

import argparse
import csv
from dataclasses import replace

import numpy as np
import torch

from nca.utils import ROOT, device, save_grid, to_uint8
from nca.world import World, WorldParams

from e1_diversity import load_evo  # noqa: E402

RES = ROOT / "experiments" / "results"


def snapshot(w):
    return to_uint8(w.x[:, :4])[0]


def sample_population(w, k, rng):
    """Yaşayan bireylerden alanla ağırlıklı k genom (E5 uygunluk eğrisi için)."""
    live = [o for o in w.orgs.values() if o.alive and o.genome]
    if not live:
        return np.zeros((0, w.model.n_genome), np.float32)
    p = np.array([o.area for o in live], float)
    idx = rng.choice(len(live), size=min(k, len(live)), replace=False, p=p / p.sum())
    return np.array([live[i].genome for i in idx], np.float32)


def run_world(model, phys, params, steps, seed, size, n_seeds, frames_every=0, genomes=None, snap_every=0, snap_k=16):
    w = World(model, phys, size=size, n_seeds=n_seeds, params=params, seed=seed, genomes=genomes)
    frames, snaps = [], {0: w_genomes0(w)}
    rng = np.random.default_rng(seed + 5)
    for t in range(steps):
        w.step()
        if frames_every and w.t % frames_every == 0:
            frames.append(snapshot(w))
        if snap_every and w.t % snap_every == 0:
            snaps[w.t] = sample_population(w, snap_k, rng)
    w.snapshots = snaps
    return w, frames


def w_genomes0(w):
    """Başlangıç (kurucu) genomları: ekilen tohumların genomu."""
    return np.array([w.g[0, :, y, x].cpu().numpy() for y, x in w.seed_positions], np.float32)


def descendants_and_ancestors(w, k):
    """Her soydan (kök ata) en büyük yaşayan torun ve o soyun atası; en fazla k çift."""
    best = {}
    for o in w.orgs.values():
        if o.alive and o.generation > 0 and o.founder in w.founder_genome:
            if o.founder not in best or o.area > best[o.founder].area:
                best[o.founder] = o
    pairs = sorted(best.values(), key=lambda o: -o.area)[:k]
    return [o.genome for o in pairs], [w.founder_genome[o.founder] for o in pairs], [o.generation for o in pairs]


def competition(model, phys, params, evolved, ancestors, steps, seed, size):
    """Torun grubu etiket grubu 0, ata grubu 1; kesmede doğan parça grubunu korur. Sonda
    hücrelerin torun grubundaki payı (soy takibine ya da genom benzerliğine dayanmaz)."""
    dev = device()
    founders = torch.tensor(np.array(evolved + ancestors), dtype=torch.float32, device=dev)
    groups = [0] * len(evolved) + [1] * len(ancestors)
    w = World(model, phys, size=size, params=replace(params, mu=0.0), seed=seed, genomes=founders, groups=groups)
    for _ in range(steps):
        w.step()
    ev, an = w.tag_group_cells()
    org_groups = [int(w.tag[0, 0][int(o.centroid[0]), int(o.centroid[1])] >= w.TAG_SPLIT) for o in w.orgs.values() if o.alive]
    return {
        "evolved_cell_frac": ev / max(1, ev + an),
        "evolved_orgs": org_groups.count(0),
        "ancestral_orgs": org_groups.count(1),
    }, snapshot(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ancestor_s0")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--k", type=int, default=6, help="rekabette grup başına kurucu sayısı")
    ap.add_argument("--comp-steps", type=int, default=15000)
    ap.add_argument("--mu", type=float, default=WorldParams.mu)
    ap.add_argument("--comp-reps", type=int, default=3, help="rekabet tekrarı (farklı yerleşim tohumları)")
    ap.add_argument("--snap-every", type=int, default=5000, help="popülasyon genom örneklemesi aralığı (E5)")
    ap.add_argument("--no-competition", action="store_true")
    ap.add_argument("--only-evo", action="store_true", help="mutasyonsuz kontrolü atla (μ taraması)")
    ap.add_argument("--cond-name", default=None, help="evrim koşulunun adı (ör. evrim_mu0.5)")
    ap.add_argument("--history", default="e3_history.csv")
    args = ap.parse_args()
    dev = device()
    model, phys, _, _ = load_evo(ROOT / "runs" / args.run, dev)
    out = RES / "e3"
    out.mkdir(parents=True, exist_ok=True)
    hist_rows, comp_rows = [], []
    for seed in args.seeds:
        conds = [(args.cond_name or "evrim", args.mu)] + ([] if args.only_evo else [("mutasyonsuz", 0.0)])
        for cond, mu in conds:
            params = WorldParams(mu=mu)
            w, frames = run_world(model, phys, params, args.steps, seed, args.size, args.n_seeds,
                                  frames_every=args.steps // 8, snap_every=args.snap_every)
            np.savez(out / f"{args.run}_{cond}_seed{seed}_snapshots.npz", **{f"t{t}": v for t, v in w.snapshots.items()})
            save_grid(out / f"{args.run}_{cond}_seed{seed}.png", frames, scale=1, ncol=8)
            for h in w.history:
                hist_rows.append({"run": args.run, "cond": cond, "seed": seed, **h})
            n_cut = sum(1 for e in w.events if e[1] == "kesme")
            print(f"[{cond} seed={seed}] t={w.t} canlı={w.history[-1]['n_org']} toplam_birey={len(w.orgs)} kesme={n_cut}", flush=True)
            if args.no_competition:
                continue
            evolved, ancestors, gens = descendants_and_ancestors(w, args.k)
            if len(evolved) < 2:
                print(f"  rekabet atlandı: yalnız {len(evolved)} soy")
                continue
            dist = float(np.mean(np.linalg.norm(np.array(evolved) - np.array(ancestors), axis=1)))
            for rep in range(args.comp_reps):
                res, img = competition(model, phys, params, evolved, ancestors, args.comp_steps, seed * 100 + rep, args.size)
                if rep == 0:
                    save_grid(out / f"{args.run}_{cond}_competition_seed{seed}.png", [img], scale=2)
                comp_rows.append({"run": args.run, "cond": cond, "seed": seed, "rep": rep, "n_lineages": len(evolved),
                                  "mean_generation": float(np.mean(gens)), "genome_distance": dist, **res})
                print(f"  rekabet: {res}", flush=True)
    for name, rows in ((args.history, hist_rows), ("e3_competition.csv", comp_rows)):
        if rows:
            with open(RES / name, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
            print(f"yazıldı: experiments/results/{name}")


if __name__ == "__main__":
    main()
