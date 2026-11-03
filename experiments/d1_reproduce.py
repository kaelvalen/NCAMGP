"""D1 - Yeniden üretim: büyüme, dairesel hasar onarımı ve 1000 adım kararlılığı.

Çıktı: experiments/results/d1.csv ve experiments/results/d1/<koşu>_*.png|gif
Dairesel hasarın merkezi rastgele bir canlı hücredir (ıskalama olmaz).
Başarı kriterleri: büyümede alfa IoU >= 0.95, dairesel hasarda >%95 onarım,
1000 adımda desenin bozulmaması.
"""

import csv

import torch

from nca import damage as dmg
from nca.evaluate import RESULTS, Ctx, base_args, find_runs
from nca.metrics import batch_iou, rgba_mse, settle_time
from nca.utils import save_gif, save_grid, set_seed, to_uint8

GROW_SNAPS = [0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300]


@torch.no_grad()
def evaluate(run_dir, n, eval_seed, out, r_range):
    ctx = Ctx(run_dir, 72)
    set_seed(eval_seed)
    gen = torch.Generator().manual_seed(eval_seed)
    name = run_dir.name

    # büyüme dizisi
    x = ctx.seed_batch(n)
    snaps, frames = [to_uint8(x[:1])[0]], []
    ious = {}
    for t in range(1, 1001):
        x = ctx.model(x)
        if t in GROW_SNAPS:
            snaps.append(to_uint8(x[:1])[0])
        if t <= 300:
            frames.append(to_uint8(x[:1])[0])
        if t in (100, 200, 300, 1000):
            ious[t] = batch_iou(x, ctx.target)
            if t == 200:
                mse200 = rgba_mse(x, ctx.target).mean().item()
    save_grid(out / f"{name}_growth.png", snaps, scale=3)
    save_gif(out / f"{name}_growth.gif", frames[::2], scale=4)
    area = (x[:, 3] > 0.1).flatten(1).sum(1).float()
    exploded = (area > 2 * ctx.target_area).float().mean().item()
    faded = (area < 0.5 * ctx.target_area).float().mean().item()

    # dairesel hasar onarımı
    x = ctx.grow(n, 300)
    before = to_uint8(x[:4])
    area0 = (x[:, 3] > 0.1).flatten(1).sum(1).float()
    dmg.circle_damage(x, gen, r_range, on_alive=True)
    removed = 1 - (x[:, 3] > 0.1).flatten(1).sum(1).float() / area0
    damaged = to_uint8(x[:4])
    series, x_end = ctx.track(x, 200)
    hit = settle_time(series, 0.9)
    after = to_uint8(x_end[:4])
    rows = [f for trip in zip(before, damaged, after) for f in trip]
    save_grid(out / f"{name}_circle_repair.png", rows, scale=3, ncol=3)

    ok = hit >= 0
    return {
        "run": name,
        "regime": ctx.name,
        "seed": ctx.seed,
        "iou_100": ious[100].mean().item(),
        "iou_200": ious[200].mean().item(),
        "iou_200_std": ious[200].std().item(),
        "mse_200": mse200,
        "iou_300": ious[300].mean().item(),
        "iou_1000": ious[1000].mean().item(),
        "iou_1000_std": ious[1000].std().item(),
        "exploded_1000": exploded,
        "faded_1000": faded,
        "circle_repair_rate": ok.float().mean().item(),
        "circle_repair_time": hit[ok].float().mean().item() if ok.any() else float("nan"),
        "circle_final_iou": series[-1].mean().item(),
        "circle_removed_frac": removed.mean().item(),
    }


def main():
    ap = base_args(__doc__, ("r?_s*",))
    ap.add_argument("--circle-r", type=float, nargs=2, default=(0.1, 0.25),
                    help="hasar yarıçapı aralığı, [-1,1] normalize (72 ızgarada 0.1 = 3.6 piksel)")
    args = ap.parse_args()
    out = RESULTS / "d1"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in find_runs(args.runs):
        row = evaluate(r, args.n, args.eval_seed, out, tuple(args.circle_r))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
        rows.append(row)
    path = RESULTS / (args.out or "d1.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: {path}")


if __name__ == "__main__":
    main()
