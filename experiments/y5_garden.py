"""Y5 ortak bahçe: evrimleşmiş genom mutasyonsuz taze dünyada da büyük/yavaş/az bölünen bedeni üretiyor mu?

Gruplar: ata (g = 0), evrim (life2l mu1 tohum başına son ortalama genom), rastgele (her evrim genomuyla aynı
normda, rastgele yönlü). Her genom kendi dünyasında: 6 kurucu aynı genomla, mu = mu_base = cut_rate = 0.
Ölçüm son --window adımın ortalaması; genom başına --reps tekrar (dünya tohumu 0..reps-1).

Kullanım: uv run python experiments/y5_garden.py [--steps 15000 --size 128 --reps 3 --jobs 6]
Çıktı: experiments/results/y5_garden.csv (satır = genom × tekrar; gid = genom sırası, tekrarlar bununla ortalanır)
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nca.life_world import LifeParams, LifeWorld  # noqa: E402
from nca.train_life import load_life  # noqa: E402
from nca.utils import ROOT, load_config  # noqa: E402


def garden_genomes(evolved, n_random, rng):
    """[(grup, kaynak tohum, genom)]: 1 ata (sıfır), k evrim, k * n_random aynı normda rastgele yönlü."""
    out = [("ata", -1, np.zeros(evolved.shape[1]))]
    out += [("evrim", k, e) for k, e in enumerate(evolved)]
    for k, e in enumerate(evolved):
        for _ in range(n_random):
            v = rng.normal(size=e.shape)
            while v @ e / (np.linalg.norm(v) * np.linalg.norm(e)) >= 0.9:  # 32 boyutta pratikte hiç dönmez
                v = rng.normal(size=e.shape)
            out.append(("rastgele", k, v * np.linalg.norm(e) / np.linalg.norm(v)))
    return out


def measure(world, hist_window):
    """Son hist_window adımdaki kayıtların ortalaması; bölünme 1000 adım başına, kapsama canlı hücre / alan."""
    h = [r for r in world.history if r["t"] > world.t - hist_window]
    m = {k: float(np.mean([r[k] for r in h])) for k in
         ("n_org", "mean_area", "speed", "pred_share", "fission_self", "cells", "resource")}
    return {"n_org": m["n_org"], "mean_area": m["mean_area"], "speed": m["speed"], "pred_share": m["pred_share"],
            "fission_self": m["fission_self"] * 1000 / world.p.track_every,
            "coverage": m["cells"] / world.size ** 2, "R": m["resource"]}


@torch.no_grad()
def run_one(job, cfg, steps, size, window):
    gid, group, src, genome, rep = job
    model, phys = load_life(ROOT / "runs" / cfg["run"])
    model.film_scale = cfg["film_scale"]
    p = LifeParams(**{**cfg["world"], "mu": 0.0, "mu_base": 0.0, "cut_rate": 0.0})
    g = torch.tensor(genome, dtype=torch.float32).repeat(6, 1)
    w = LifeWorld(model, phys, size=size, n_seeds=6, params=p, seed=rep, genomes=g)
    torch.manual_seed(rep)
    t0 = time.time()
    while w.t < steps and not w.extinct:
        w.step()
    return {"gid": gid, "group": group, "src": src, "rep": rep, "norm": float(np.linalg.norm(genome)), "t": w.t,
            "extinct": bool(w.extinct), **measure(w, window), "sec": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/life_world.yaml")
    ap.add_argument("--ref", default="life2l", help="evrim genomlarının alındığı koşular (<ref>_mu1_s*)")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n_random", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="experiments/results/y5_garden.csv")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config, [])
    gcols = [f"g{i}" for i in range(32)]
    runs = sorted((ROOT / "runs").glob(f"{args.ref}_mu1_s*"))
    evolved = np.array([pd.read_csv(r / "history.csv")[gcols].values[-1] for r in runs])
    genomes = garden_genomes(evolved, args.n_random, np.random.default_rng(0))
    jobs = [(k, grp, src, g, rep) for k, (grp, src, g) in enumerate(genomes) for rep in range(args.reps)]
    print(f"{len(genomes)} genom × {args.reps} tekrar = {len(jobs)} dünya, {args.jobs} süreç", flush=True)
    rows, out = [], ROOT / args.out
    with ProcessPoolExecutor(args.jobs, mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(run_one, j, cfg, args.steps, args.size, args.window) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
            with open(out, "w", newline="") as fh:  # her sonuçta yeniden yaz: yarıda kesilse de veri kalır
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
