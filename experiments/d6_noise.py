"""D6 - Gürültüden başlatma (esnek): gürültüden kaç birey çıkıyor?

Her model için n gürültü başlangıcından T adım çalıştırılır; birey sayısı dağılımı ve
bireylerin hedefle hizalanmış IoU'su raporlanır. Varsayılan olarak hem gürültü modeli
(noise_s*) hem de tohumdan eğitilmiş modeller (karşılaştırma) değerlendirilir.

Çıktı: experiments/results/d6.csv
"""

import csv
from collections import Counter

import numpy as np
import torch

from nca.evaluate import RESULTS, Ctx, base_args, find_runs, to_np
from nca.metrics import component_ious
from nca.train_noise import noise_state
from nca.utils import load_target, save_grid, set_seed, to_uint8


@torch.no_grad()
def main():
    ap = base_args(__doc__, ("noise_s*", "r3_s0"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--density", type=float, default=0.003, help="configs/noise.yaml ile aynı olmalı")
    args = ap.parse_args()
    out = RESULTS / "d6"
    out.mkdir(parents=True, exist_ok=True)
    tmask = (load_target()["rgba"][3] > 0.1).numpy()
    rows = []
    for r in find_runs(args.runs):
        set_seed(args.eval_seed)
        gen = torch.Generator().manual_seed(args.eval_seed)
        ctx = Ctx(r, args.grid)
        x = noise_state(args.n, args.grid, ctx.model.channels, args.density, ctx.dev, gen)
        x = ctx.model.rollout(x, args.steps)
        counts, ious = [], []
        for i in range(args.n):
            cis = component_ious(to_np(x[i])[..., 3] > 0.1, tmask)
            counts.append(len(cis))
            ious += cis
        save_grid(out / f"{r.name}_noise.png", to_uint8(x[:12]), scale=2, ncol=6)
        dist = Counter(counts)
        row = {
            "run": r.name,
            "count_mean": float(np.mean(counts)),
            "count_std": float(np.std(counts)),
            "count_dist": " ".join(f"{k}:{v}" for k, v in sorted(dist.items())),
            "individual_iou_mean": float(np.mean(ious)) if ious else 0.0,
            "good_individuals_per_sample": float(np.sum(np.array(ious) >= 0.8) / args.n) if ious else 0.0,
        }
        rows.append(row)
        print(row)
    path = RESULTS / (args.out or "d6.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: {path}")


if __name__ == "__main__":
    main()
