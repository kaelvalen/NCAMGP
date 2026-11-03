"""D2 - Parça boyutu-yenilenme başarısı eğrisi (makalenin ana şekli).

İki mod:
  inplace  72x72 ızgarada rastgele açılı yarım düzlem kesmesi; kalan parça eksik tarafı
           orijinal konumunda büyütmeli. Başarı: 200. adımda IoU >= 0.9 (süre: eşiğin üstünde kalmaya başladığı adım).
  split    128x128 ızgarada organizma ikiye kesilir, parçalar `gap` piksel uzaklaştırılır.
           Her parçanın kendi başına tam bir bireye dönüşüp dönüşmediği (hizalanmış IoU)
           ve birey sayısı ölçülür. Kesme eğitimi olmadan bölünmenin kendiliğinden çıkıp
           çıkmadığı sorusu (katkı 2) bu moddan okunur.

Çıktı: experiments/results/d2_<mod>.csv (koşu x parça oranı başına bir satır)
"""

import csv
import math

import numpy as np
import torch

from nca import damage as dmg
from nca.evaluate import RESULTS, Ctx, base_args, find_runs, to_np
from nca.metrics import aligned_iou, component_ious, components, settle_time
from nca.utils import save_grid, set_seed, to_uint8

FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@torch.no_grad()
def inplace(ctx, n, gen, steps, thr, out):
    grown = ctx.grow(n, 300)
    rows, examples = [], []
    for f in FRACS:
        x = grown.clone()
        for i in range(n):
            dmg.halfplane_cut(x[i], float(torch.rand(1, generator=gen)) * 2 * math.pi, f)
        cut_img = to_uint8(x[:3])
        series, x_end = ctx.track(x, steps)
        hit = settle_time(series, thr)
        ok = hit >= 0
        al = [aligned_iou(to_np(x_end[i])[..., 3] > 0.1, ctx.target_mask) for i in range(n)]
        examples += [f for pair in zip(cut_img, to_uint8(x_end[:3])) for f in pair]
        rows.append(
            {
                "keep_frac": f,
                "success": ok.float().mean().item(),
                "success_aligned": float(np.mean(np.array(al) >= thr)),
                "time_mean": hit[ok].float().mean().item() if ok.any() else float("nan"),
                "final_iou": series[-1].mean().item(),
                "final_iou_std": series[-1].std().item(),
                "final_aligned_iou": float(np.mean(al)),
            }
        )
    save_grid(out / f"{ctx.name}_s{ctx.seed}_inplace.png", examples, scale=2, ncol=6)
    return rows


def piece_ious(mask, target_mask, angle, size):
    """Bileşenleri ağırlık merkezlerinin kesme normali üzerindeki izdüşümüne göre parçalara
    eşler. (A parçası = normal tarafı, B parçası = karşı taraf) için hizalanmış IoU döndürür;
    karşılığı olmayan parçanın IoU'su 0'dır."""
    comps, lab = components(mask)
    c = (size - 1) / 2
    proj = [(r.centroid[1] - c) * math.cos(angle) + (r.centroid[0] - c) * math.sin(angle) for r in comps]
    cis = component_ious(mask, target_mask)
    if not comps:
        return 0.0, 0.0, 0
    order = np.argsort(proj)
    a = cis[order[-1]] if proj[order[-1]] >= 0 or len(comps) > 1 else 0.0
    b = cis[order[0]] if proj[order[0]] < 0 or len(comps) > 1 else 0.0
    return a, b, len(comps)


@torch.no_grad()
def split(ctx, n, gen, steps, thr, gap, out):
    """f <= 0.5 için kesilir; A parçası f, B parçası 1-f oranındadır. Parça başı satırlar
    piece_frac'a göre toplanır, kesme başı istatistikler cut_frac'a göre."""
    grown = ctx.grow(n, 300)
    per_piece = {f: [] for f in FRACS}
    rows, examples = [], []
    for f in [v for v in FRACS if v <= 0.5]:
        angles = [float(torch.rand(1, generator=gen)) * 2 * math.pi for _ in range(n)]
        x = torch.stack([dmg.split_and_separate(grown[i], angles[i], f, gap) for i in range(n)])
        cut_img = to_uint8(x[:3])
        x_end = ctx.model.rollout(x, steps)
        n_ind, both = [], []
        for i in range(n):
            mask = to_np(x_end[i])[..., 3] > 0.1
            a, b, k = piece_ious(mask, ctx.target_mask, angles[i], ctx.size)
            per_piece[f].append(a)
            per_piece[round(1 - f, 1)].append(b)
            n_ind.append(k)
            both.append(k == 2 and a >= thr and b >= thr)
        examples += [f_ for pair in zip(cut_img, to_uint8(x_end[:3])) for f_ in pair]
        rows.append(
            {
                "kind": "cut",
                "frac": f,
                "n_individuals_mean": float(np.mean(n_ind)),
                "two_individuals": float(np.mean(np.array(n_ind) == 2)),
                "both_regenerated": float(np.mean(both)),
                "piece_iou_mean": float("nan"),
                "piece_regenerated": float("nan"),
            }
        )
    for f, vals in per_piece.items():
        if vals:
            v = np.array(vals)
            rows.append(
                {
                    "kind": "piece",
                    "frac": f,
                    "n_individuals_mean": float("nan"),
                    "two_individuals": float("nan"),
                    "both_regenerated": float("nan"),
                    "piece_iou_mean": float(v.mean()),
                    "piece_regenerated": float((v >= thr).mean()),
                }
            )
    save_grid(out / f"{ctx.name}_s{ctx.seed}_split.png", examples, scale=1, ncol=6)
    return rows


def main():
    ap = base_args(__doc__)
    ap.add_argument("--mode", choices=["inplace", "split"], default="inplace")
    ap.add_argument("--steps", type=int, default=None, help="inplace: 200, split: 400")
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--gap", type=int, default=44, help="toplam ayırma; organizma boyundan (36 px) büyük olmalı, yoksa enine kesmede yeniden büyüyen parçalar birleşir")
    args = ap.parse_args()
    out = RESULTS / "d2"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in find_runs(args.runs):
        set_seed(args.eval_seed)
        gen = torch.Generator().manual_seed(args.eval_seed)
        if args.mode == "inplace":
            ctx = Ctx(r, 72)
            res = inplace(ctx, args.n, gen, args.steps or 200, args.thr, out)
        else:
            ctx = Ctx(r, 128)
            res = split(ctx, args.n, gen, args.steps or 400, args.thr, args.gap, out)
        for row in res:
            row = {"run": r.name, "regime": ctx.name, "seed": ctx.seed, **row}
            rows.append(row)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    path = RESULTS / (args.out or f"d2_{args.mode}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: {path}")


if __name__ == "__main__":
    main()
