"""FiLM ölçeği (σ_P) kalibrasyonu: genom kuralı ne kadar değiştirsin? Her ölçek için rastgele
genomlu (g ~ N(0, I)) n canlı tek hücreden büyür; 2000. adımda yaşayan oranı, ortalama kütle ve
g = 0 atasına göre beden betimleyici mesafesi ölçülür. Hedef: yaşayan oranı ~0,5 ve mesafe > 0.
Ayrıca 192² LifeWorld için saniye/adım ölçülür (koşu süresi tahmini).

Çıktı: experiments/results/film_scale.csv (scale, survive_frac, mean_mass, desc_dist)
Kullanım: uv run python experiments/calibrate_film.py --run life_s0 [--scales 0.25 0.5 1 2 4]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nca.evo import make_seed  # noqa: E402
from nca.life import M_CH, life_step  # noqa: E402
from nca.life_world import LifeParams, LifeWorld  # noqa: E402
from nca.train_life import body_descriptors, center_torus, load_life  # noqa: E402
from nca.utils import ROOT, load_config  # noqa: E402
from y1_viability import bodies  # noqa: E402


@torch.no_grad()
def grow(model, phys, genomes, size, steps):
    x, g = make_seed(genomes, size, model.channels)
    R = torch.ones(len(genomes), 1, size, size, device=genomes.device)
    for _ in range(steps):
        x, g, R, _, _ = life_step(model, x, g, R, phys)
    return center_torus(x, R)[0] if phys.torus else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="life_s0")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--time_steps", type=int, default=500)
    ap.add_argument("--out", default="film_scale.csv")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    model, phys = load_life(run)
    cfg = load_config(run / "config.yaml")
    dev = next(model.parameters()).device
    size, mv = cfg["grid"], cfg["min_viable"]
    torch.manual_seed(0)
    genomes = torch.randn(args.n, model.n_genome, device=dev)
    model.film_scale = 0.0
    anc = body_descriptors(grow(model, phys, torch.zeros(1, model.n_genome, device=dev), size, args.steps))[0]
    rows = []
    for s in args.scales:
        model.film_scale = s
        x = grow(model, phys, genomes, size, args.steps)
        m = x[:, M_CH].cpu().numpy()
        alive = np.array([bodies(mi, mv) > 0 for mi in m])
        d = (body_descriptors(x) - anc).norm(dim=1).cpu().numpy()
        rows.append({"scale": s, "survive_frac": round(float(alive.mean()), 3),
                     "mean_mass": round(float(x[:, M_CH].flatten(1).sum(1).mean()), 1),
                     "desc_dist": round(float(d[alive].mean()) if alive.any() else 0.0, 3)})
        print(rows[-1], flush=True)
    path = ROOT / "experiments" / "results" / args.out
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    model.film_scale = 1.0
    w = LifeWorld(model, phys, size=192, n_seeds=12, params=LifeParams(), seed=0)
    for _ in range(50):  # ısınma
        w.step()
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.time()
    for _ in range(args.time_steps):
        w.step()
    torch.cuda.synchronize() if dev.type == "cuda" else None
    sps = (time.time() - t0) / args.time_steps
    print(f"LifeWorld 192²: {sps * 1000:.1f} ms/adım -> 100k adım {sps * 1e5 / 3600:.1f} saat -> {path}")


if __name__ == "__main__":
    main()
