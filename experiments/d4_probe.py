"""D4 - Doğrusal prob: gizli kanallar baş-kuyruk konumunu kodluyor mu? (H3: R^2 > 0.8)

Büyümüş organizmalardaki canlı hücrelerin gizli kanallarından, hücrenin hedefteki eksen
konumu (baş ucu 0, kuyruk ucu 1) Ridge regresyonla tahmin edilir. 5 katlı çapraz doğrulama
R^2'si raporlanır: tüm gizli kanallar, tek tek kanallar, açgözlü ileri seçimle k kanal ve
karşılaştırma için 2. derece polinom prob ve yalnız RGBA (renk zaten baş/kuyruğu ayırt ettiğinden üst sınır değil
taban çizgisi olarak okunmalı).

Doğrusal okunabilirlik nedensellik değildir; D5 müdahale deneyi bu yüzden gerekli.

Çıktı: runs/<koşu>/probe.json (D5 ve demo bunu kullanır), experiments/results/d4.csv
"""

import csv

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from nca.evaluate import RESULTS, Ctx, base_args, find_runs
from nca.utils import set_seed, write_json


def cv_r2(X, y, groups, model=None):
    model = model or Ridge(alpha=1e-2)
    return float(cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups, scoring="r2").mean())


def quad_probe():
    """Karşılaştırma: 2. derece polinom özelliklerle Ridge (bilgi var ama doğrusal değil mi?)."""
    return make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0))


@torch.no_grad()
def collect(ctx, n, times=(300, 350, 400)):
    x = ctx.grow(n, times[0])
    snaps = [x]
    for a, b in zip(times, times[1:]):
        x = ctx.model.rollout(x, b - a)
        snaps.append(x)
    inside = ctx.axial >= 0
    feats, axial, lateral, region, groups = [], [], [], [], []
    for k, s in enumerate(snaps):
        s = s.permute(0, 2, 3, 1).cpu().numpy()
        for i in range(n):
            m = inside & (ctx.region > 0) & (s[i, ..., 3] > 0.1)
            feats.append(s[i][m])
            axial.append(ctx.axial[m])
            region.append(ctx.region[m])
            groups.append(np.full(m.sum(), i))  # aynı organizmanın pikselleri aynı katta
    return np.concatenate(feats), np.concatenate(axial), np.concatenate(region), np.concatenate(groups)


def probe(ctx, n):
    F, y, region, groups = collect(ctx, n)
    H = F[:, 4:]
    nh = H.shape[1]
    res = {
        "r2_hidden_all": cv_r2(H, y, groups),
        "r2_rgba": cv_r2(F[:, :4], y, groups),
        "r2_all": cv_r2(F, y, groups),
        "r2_hidden_quad": cv_r2(H, y, groups, quad_probe()),
        "r2_single": [cv_r2(H[:, [c]], y, groups) for c in range(nh)],
    }
    chosen, greedy = [], []
    for _ in range(nh):
        best = max((c for c in range(nh) if c not in chosen), key=lambda c: cv_r2(H[:, chosen + [c]], y, groups))
        chosen.append(best)
        greedy.append(cv_r2(H[:, chosen], y, groups))
    res["greedy_order"] = chosen  # gizli kanal indeksi (durumda 4 + i)
    res["greedy_r2"] = greedy
    mu, sd = H.mean(0), H.std(0) + 1e-8
    reg = Ridge(alpha=1e-2).fit((H - mu) / sd, y)
    res["coef_standardized"] = reg.coef_.tolist()
    for name, rid in [("head", 1), ("body", 2), ("tail", 3)]:
        res[f"{name}_means"] = H[region == rid].mean(0).tolist()
    res["n_pixels"] = int(len(y))
    return res


def main():
    ap = base_args(__doc__)
    ap.set_defaults(n=32)
    args = ap.parse_args()
    rows = []
    for r in find_runs(args.runs):
        set_seed(args.eval_seed)
        ctx = Ctx(r, 72)
        res = probe(ctx, args.n)
        write_json(r / "probe.json", res)
        k3 = res["greedy_r2"][min(2, len(res["greedy_r2"]) - 1)]
        row = {
            "run": r.name,
            "regime": ctx.name,
            "seed": ctx.seed,
            "r2_hidden_all": res["r2_hidden_all"],
            "r2_rgba": res["r2_rgba"],
            "r2_all": res["r2_all"],
            "r2_hidden_quad": res["r2_hidden_quad"],
            "r2_best_single": max(res["r2_single"]),
            "best_single_channel": int(np.argmax(res["r2_single"])),
            "r2_greedy_3": k3,
            "greedy_top3": " ".join(map(str, res["greedy_order"][:3])),
        }
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    path = RESULTS / (args.out or "d4.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: {path}")


if __name__ == "__main__":
    main()
