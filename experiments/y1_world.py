"""Y1-dünya: ata büyük sarmal dünyada uzun ufukta sınırlı kalıyor mu? Kesme ve mutasyon yok; yalnız ata.

Ölçülen (her --every adımda): birey sayısı, ortalama alan, canlı hücre kapsaması, ortalama R.
Geçme ölçütü (2026-11-27, sonuçlar görülmeden yazıldı): her r0 için son adımda birey > 0 ve
kapsama < %40 (ilk ata life_s0: r0 = 1'de 20k adımda büyüme sürüyordu, r0 <= 0,35'te hiç büyümedi).

Kullanım: uv run python experiments/y1_world.py --run life2_s0 [--r0 0.3 1.0 --steps 30000]
Çıktı: experiments/results/y1_world_<run>.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nca.life import M_CH  # noqa: E402
from nca.life_world import LifeParams, LifeWorld  # noqa: E402
from nca.train_life import load_life  # noqa: E402
from nca.utils import ROOT  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="life2_s0")
    ap.add_argument("--r0", type=float, nargs="+", default=[0.3, 1.0])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--size", type=int, default=192)
    args = ap.parse_args()
    model, phys = load_life(ROOT / "runs" / args.run)
    model.film_scale = 1.0  # g = 0: etkisiz, dünyadakiyle aynı model
    rows = []
    for r0 in args.r0:
        torch.manual_seed(0)
        w = LifeWorld(model, phys, size=args.size, n_seeds=12, params=LifeParams(mu=0.0, cut_rate=0.0), seed=0)
        w.R *= r0
        for s in range(1, args.steps + 1):
            w.step()
            if s % args.every == 0 or w.extinct:
                live = w.live_orgs()
                a = w.x[0, M_CH] > 0.1
                rows.append({"r0": r0, "t": s, "n_org": len(live),
                             "mean_area": round(sum(o.area for o in live) / max(len(live), 1), 1),
                             "coverage": round(float(a.float().mean()), 4), "R_mean": round(float(w.R.mean()), 4)})
                print(rows[-1], flush=True)
                if w.extinct:
                    break
        last = rows[-1]
        ok = last["n_org"] > 0 and last["coverage"] < 0.4
        print(f"[{args.run}] r0={r0}: {'GEÇTİ' if ok else 'KALDI'} (birey {last['n_org']}, kapsama %{100 * last['coverage']:.0f})")
    path = ROOT / "experiments" / "results" / f"y1_world_{args.run}.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("->", path)


if __name__ == "__main__":
    main()
