"""Yaşayan ekosistem: koşul başına zaman serileri ve karar noktası özeti.

Girdi: runs/lifeworld_<koşul>_s<tohum>/history.csv, summary.yaml (experiments/y_run.py)
Çıktı: analysis/figures/life_timeseries.png|pdf, analysis/summary_life.md

Kullanım: (cd analysis && uv run python plots_life.py [--name lifeworld])
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from plots import INK2, ROOT, save  # noqa: E402

COND = {"mu1": "#2a78d6", "mu0": "#eb6834"}
COND_LABEL = {"mu1": "evrim (μ > 0)", "mu0": "mutasyonsuz (μ = 0)"}
PANELS = [("n_org", "canlı birey"), ("speed", "hız (hücre / adım)"), ("pred_share", "avcılık payı"),
          ("fission_self", "kendiliğinden bölünme / 1000 adım")]
SMOOTH = 50  # kayıt (track_every = 10 -> 500 adım)


def load(name):
    runs = []
    for d in sorted((ROOT / "runs").glob(f"{name}_mu*_s*")):
        if not (d / "history.csv").exists():
            continue
        h = pd.read_csv(d / "history.csv")
        s = yaml.safe_load((d / "summary.yaml").read_text())
        cond, seed = d.name[len(name) + 1 :].rsplit("_s", 1)
        dt = s["params"]["track_every"]
        h["fission_self"] = h["fission_self"] * 1000 / dt  # pencere başına sayı -> 1000 adım başına
        runs.append({"cond": cond, "seed": int(seed), "h": h, "s": s})
    return runs


def fig_timeseries(runs, anneal):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7, 2.0 * len(PANELS)), sharex=True)
    for ax, (col, label) in zip(axes, PANELS):
        for cond, color in COND.items():
            rs = [r["h"].set_index("t")[col].rolling(SMOOTH, min_periods=1).mean() for r in runs if r["cond"] == cond]
            if not rs:
                continue
            df = pd.concat(rs, axis=1)
            mean, se = df.mean(1), df.std(1).fillna(0) / np.sqrt(df.notna().sum(1))
            ax.plot(df.index, mean, color=color, lw=1.4, label=f"{COND_LABEL[cond]} (n = {len(rs)})")
            ax.fill_between(df.index, mean - se, mean + se, color=color, alpha=0.2, lw=0)
        ax.axvline(anneal, color=INK2, ls="--", lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(anneal, axes[0].get_ylim()[1], " dış kesmeler sıfır", color=INK2, fontsize=8, va="top")
    axes[-1].set_xlabel("adım")
    save(fig, "life_timeseries")


def window(h, lo, hi):
    w = h[(h["t"] > lo) & (h["t"] <= hi)]
    return {"n_org": w["n_org"].mean(), "speed": w["speed"].mean(), "pred_share": w["pred_share"].mean(),
            "self_per_1k": w["fission_self"].mean()}


def summary(runs, anneal):
    lines = ["# Yaşayan ekosistem — karar noktası özeti\n",
             f"Erken pencere: ilk 5000 adım (ataya yakın). Geç pencere: kesmeler söndükten (t > {anneal}) sonraki",
             "son %20. Kendiliğinden bölünme 1000 adım başına; avcılık payı = yemeyle alınan / (yeme + kaynak alımı).\n",
             "| koşul | tohum | son adım | tükendi | birey erken → geç | hız erken → geç | avcılık erken → geç | kendiliğinden bölünme erken → geç |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["cond"], r["seed"])):
        h, t_end = r["h"], int(r["h"]["t"].max())
        e = window(h, 0, 5000)
        lo = max(anneal, t_end - 0.2 * t_end)
        g = window(h, lo, t_end) if t_end > anneal else {k: float("nan") for k in e}
        lines.append(f"| {r['cond']} | {r['seed']} | {t_end} | {'evet' if r['s']['extinct'] else 'hayır'} | "
                     + " | ".join(f"{e[k]:.3g} → {g[k]:.3g}" for k in e) + " |")
    out = ROOT / "analysis" / "summary_life.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"özet: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lifeworld")
    args = ap.parse_args()
    runs = load(args.name)
    if not runs:
        raise SystemExit(f"koşu yok: runs/{args.name}_mu*_s*")
    anneal = runs[0]["s"]["params"]["anneal_steps"]
    fig_timeseries(runs, anneal)
    summary(runs, anneal)


if __name__ == "__main__":
    main()
