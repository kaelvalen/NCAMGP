"""D3 - Kesim yönü: baş tarafı, kuyruk tarafı ve orta parçadan yenilenme.

Baş-kuyruk ekseni dikeydir (baş yukarıda). Kesmeler eksene diktir; açıya +-10 derece
rastgele sapma eklenir. Başarı: 200. adımda yerinde IoU >= 0.9; ayrıca sonuçta
baş sayısı 1 olan örneklerin oranı raporlanır.

Çıktı: experiments/results/d3.csv
"""

import csv
import math

import numpy as np
import torch

from nca import damage as dmg
from nca.evaluate import RESULTS, Ctx, base_args, find_runs, to_np
from nca.metrics import count_heads, settle_time
from nca.utils import save_grid, set_seed, to_uint8

UP, DOWN = -math.pi / 2, math.pi / 2  # normal yukarı: baş tarafı kalır; aşağı: kuyruk tarafı

FRAGMENTS = [
    ("bas_tarafi", 0.3),
    ("bas_tarafi", 0.5),
    ("bas_tarafi", 0.7),
    ("kuyruk_tarafi", 0.3),
    ("kuyruk_tarafi", 0.5),
    ("kuyruk_tarafi", 0.7),
    ("orta", (0.3, 0.7)),
    ("orta", (0.35, 0.65)),
]


def cut(x, kind, frac, jitter):
    if kind == "bas_tarafi":
        dmg.halfplane_cut(x, UP + jitter, frac)
    elif kind == "kuyruk_tarafi":
        dmg.halfplane_cut(x, DOWN + jitter, frac)
    else:
        dmg.band_cut(x, DOWN + jitter, *frac)


@torch.no_grad()
def evaluate(ctx, n, gen, steps, out):
    grown = ctx.grow(n, 300)
    rows, examples = [], []
    for kind, frac in FRAGMENTS:
        x = grown.clone()
        for i in range(n):
            cut(x[i], kind, frac, (float(torch.rand(1, generator=gen)) - 0.5) * math.radians(20))
        start = to_uint8(x[:2])
        series, x_end = ctx.track(x, steps)
        hit = settle_time(series, 0.9)
        ok = hit >= 0
        heads = [count_heads(to_np(x_end[i]), ctx.palette) for i in range(n)]
        examples += [f for pair in zip(start, to_uint8(x_end[:2])) for f in pair]
        keep = frac if isinstance(frac, float) else frac[1] - frac[0]
        rows.append(
            {
                "fragment": kind,
                "keep_frac": round(keep, 2),
                "success": ok.float().mean().item(),
                "time_mean": hit[ok].float().mean().item() if ok.any() else float("nan"),
                "final_iou": series[-1].mean().item(),
                "one_head": float(np.mean(np.array(heads) == 1)),
            }
        )
    save_grid(out / f"{ctx.name}_s{ctx.seed}_direction.png", examples, scale=2, ncol=8)
    return rows


def main():
    ap = base_args(__doc__)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    out = RESULTS / "d3"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in find_runs(args.runs):
        set_seed(args.eval_seed)
        gen = torch.Generator().manual_seed(args.eval_seed)
        ctx = Ctx(r, 72)
        for row in evaluate(ctx, args.n, gen, args.steps, out):
            row = {"run": r.name, "regime": ctx.name, "seed": ctx.seed, **row}
            rows.append(row)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    path = RESULTS / (args.out or "d3.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: {path}")


if __name__ == "__main__":
    main()
