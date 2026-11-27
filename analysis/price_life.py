"""Price ayrıştırması: nüfus ortalama genomunun değişimi = birey düzeyi seçilim + aktarım.

Ardışık dökümler (t, t+T) arasında birim t'deki bireylerdir. t+T'deki her birey ata zincirinden
t'deki atasına bağlanır (kendisi de olabilir). w_i = i'nin t+T'deki torun sayısı, g'_i torunlarının
ortalama genomu. Δḡ = Cov(w, g)/w̄ (seçilim: farklı üreme ve hayatta kalma)
                    + E[w (g' - g)]/w̄ (aktarım: yaşam içi değişim = mutasyon + beden içi hücre seçilimi).
Terimler, bağımsız koşulların (--ref) son ortalama genomundan alınan ortak yöne izdüşürülür.

Kullanım: uv run python analysis/price_life.py --name life2l --ref life2
Çıktı: analysis/price_<name>.csv ve özet tablo (stdout).
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_orgs(run):
    p = run / "orgs.json"
    raw = p.read_text() if p.exists() else gzip.open(run / "orgs.json.gz", "rt").read()
    return {o["id"]: o["parent"] for o in json.loads(raw)}


def price(L, parent):
    """L: (n, 3 + G) [t, id, alan, g...]. Her ardışık döküm çifti için (t, seçilim, aktarım, bağlanamayan oranı, ḡ, ḡ')."""
    ts = np.unique(L[:, 0])
    rows = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        a, b = L[L[:, 0] == t0], L[L[:, 0] == t1]
        idx = {int(i): k for k, i in enumerate(a[:, 1])}
        g = a[:, 3:].astype(np.float64)
        w = np.zeros(len(a))
        gsum = np.zeros_like(g)
        lost = 0
        for row in b:
            j = int(row[1])
            while j >= 0 and j not in idx:
                j = parent.get(j, -1)
            if j < 0:
                lost += 1
                continue
            w[idx[j]] += 1
            gsum[idx[j]] += row[3:]
        wb = w.mean()
        sel = ((w / wb - 1)[:, None] * g).mean(0)
        has = w > 0
        tr = (gsum[has] - w[has, None] * g[has]).sum(0) / w.sum()
        rows.append((t1, sel, tr, lost / len(b), g.mean(0), gsum.sum(0) / w.sum()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="life2l")
    ap.add_argument("--ref", default="life2", help="ortak yönü veren bağımsız koşular")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    gcols = [f"g{i}" for i in range(32)]
    ref = np.array([pd.read_csv(ROOT / f"runs/{args.ref}_mu1_s{s}/history.csv")[gcols].values[-1] for s in range(4)])
    u = ref.mean(0) / np.linalg.norm(ref.mean(0))
    out = []
    for s in range(args.seeds):
        run = ROOT / f"runs/{args.name}_mu1_s{s}"
        for t, sel, tr, lost, gm0, gm1 in price(np.load(run / "lineage.npy"), load_orgs(run)):
            # Price özdeşliği: bağlanan torunların ortalaması - ata ortalaması = seçilim + aktarım
            assert np.allclose(gm1 - gm0, sel + tr, atol=1e-6)
            out.append({"seed": s, "t": int(t), "sel_u": sel @ u, "tr_u": tr @ u,
                        "sel_norm": np.linalg.norm(sel), "tr_norm": np.linalg.norm(tr), "lost": lost})
    df = pd.DataFrame(out)
    df.to_csv(ROOT / f"analysis/price_{args.name}.csv", index=False)
    print("tohum | Σ seçilim·u | Σ aktarım·u | seçilim payı | ortalama bağlanamayan")
    for s, d in df.groupby("seed"):
        S, T = d.sel_u.sum(), d.tr_u.sum()
        print(f"{s} | {S:.3f} | {T:.3f} | {S / (S + T):.2f} | {d.lost.mean():.3f}")
    for lo, hi in [(0, 50000), (50000, 10**9)]:
        d = df[(df.t > lo) & (df.t <= hi)].groupby("seed")[["sel_u", "tr_u"]].sum()
        print(f"t ∈ ({lo}, {hi}]: seçilim·u ort {d.sel_u.mean():.3f} ± {d.sel_u.std():.3f}, "
              f"aktarım·u ort {d.tr_u.mean():.3f} ± {d.tr_u.std():.3f}")


if __name__ == "__main__":
    main()
