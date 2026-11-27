"""Y1: ata ("en basit canlı") yaşayabilir mi? Tek hücreden büyür, 1000. adımda yarım düzlem kesmesi,
2000. adımda ölçüm. Çıktı: experiments/results/y1.csv

  alive_2000        2000. adımda min_viable'dan büyük en az bir beden var mı
  n_bodies          min_viable'dan büyük beden sayısı
  mass              toplam kütle
  regrew_after_cut  kesmeden sonra kütle, kesme öncesinin en az %80'ine döndü mü

Kullanım: uv run python experiments/y1_viability.py --run life_s0 [--n 64 --steps 2000 --r0 1.0]
(--r0: başlangıç besin düzeyi; kıtlık sınaması için < 1)
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nca import damage as dmg  # noqa: E402
from nca.life import M_CH, life_step  # noqa: E402
from nca.model import ALIVE_THRESHOLD  # noqa: E402
from nca.train_life import center_torus, load_life, seed_state  # noqa: E402
from nca.utils import ROOT, load_config  # noqa: E402


def bodies(m, min_viable):
    """(H, W) kütle -> min_viable'dan büyük bileşen sayısı."""
    lab, n = ndimage.label(m > ALIVE_THRESHOLD, structure=np.ones((3, 3)))
    if not n:
        return 0
    return int((ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) >= min_viable).sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="life_s0")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="y1.csv")
    ap.add_argument("--r0", type=float, default=1.0)
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    model, phys = load_life(run)
    cfg = load_config(run / "config.yaml")
    dev = next(model.parameters()).device
    torch.manual_seed(0)
    x, g, R = seed_state(model, args.n, cfg["grid"], dev, torch.full((args.n,), args.r0))
    cut_at = args.steps // 2
    before = None
    for s in range(args.steps):
        if s == cut_at:
            if phys.torus:  # sarmal arena: kesme ve ölçüm ortalanmış tek bedende
                x, R = center_torus(x, R)
            before = x[:, M_CH].flatten(1).sum(1).clone()
            gen = torch.Generator().manual_seed(1)
            for i in range(args.n):
                dmg.halfplane_cut(x[i], float(torch.rand(1, generator=gen)) * 2 * math.pi, 0.5)
            g = g * (x[:, M_CH : M_CH + 1] > 0)
        x, g, R, _, _ = life_step(model, x, g, R, phys)
    if phys.torus:
        x, R = center_torus(x, R)
    m = x[:, M_CH].cpu().numpy()
    mass = x[:, M_CH].flatten(1).sum(1)
    rows = []
    for i in range(args.n):
        nb = bodies(m[i], cfg["min_viable"])
        rows.append({"sample": i, "alive_2000": int(nb > 0), "n_bodies": nb, "mass": round(float(mass[i]), 2),
                     "regrew_after_cut": int(bool(mass[i] >= 0.8 * before[i]) and nb > 0)})
    path = ROOT / "experiments" / "results" / args.out
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = np.mean([r["alive_2000"] for r in rows])
    rg = np.mean([r["regrew_after_cut"] for r in rows])
    nb = np.mean([r["n_bodies"] for r in rows])
    print(f"[{args.run} r0={args.r0} adım={args.steps}] alive_2000={a:.2f} regrew_after_cut={rg:.2f} n_bodies={nb:.2f} "
          f"mass={float(mass.mean()):.0f} -> {path}")


if __name__ == "__main__":
    main()
