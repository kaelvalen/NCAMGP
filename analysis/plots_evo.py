"""Evrimsel katmanın şekilleri ve özet tablosu.

Girdi: experiments/results/e1.csv, e4.csv, e3_history.csv, e3_competition.csv
Çıktı: analysis/figures/e3_*.png|pdf, analysis/summary_evo.md

Kullanım: python analysis/plots_evo.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from plots import FIG, INK, INK2, RES, ROOT, save  # noqa: E402

COND = {"evrim": "#2a78d6", "mutasyonsuz": "#eb6834"}
COND_LABEL = {"evrim": "evrim (μ > 0)", "mutasyonsuz": "mutasyonsuz (μ = 0)"}


def table(df, title):
    lines = ["| " + " | ".join(df.columns) + " |", "|---" * len(df.columns) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return f"## {title}\n\n" + "\n".join(lines) + "\n"


def fig_history(h):
    h = h.copy()
    h["mean_area"] = h.get("mean_area", h["cells"] / h["n_org"].clip(lower=1))
    metrics = [("n_org", "canlı birey"), ("mean_area", "ortalama beden (hücre)"), ("genome_div", "genom çeşitliliği")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (m, lab) in zip(axes, metrics):
        for cond, col in COND.items():
            d = h[h.cond == cond]
            if d.empty:
                continue
            g = d.groupby("t")[m].agg(["mean", "std"]).reset_index().fillna(0)
            g["mean"] = g["mean"].rolling(20, min_periods=1).mean()
            ax.plot(g["t"], g["mean"], color=col, lw=1.6, label=COND_LABEL[cond])
        ax.set_title(lab, fontsize=9, color=INK)
        ax.set_xlabel("adım")
    axes[0].legend(fontsize=8, loc="lower right")
    save(fig, "e3_history")


def fig_competition(c):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for i, (cond, col) in enumerate(COND.items()):
        v = c[c.cond == cond]["evolved_cell_frac"].values
        if not len(v):
            continue
        jitter = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.25
        ax.scatter(np.full(len(v), i) + jitter, v, s=40, color=col, edgecolor="white", linewidth=1, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK, lw=2, zorder=4)
        ax.annotate(f"{v.mean():.2f} ± {v.std():.2f}", (i + 0.28, v.mean()), fontsize=8, color=INK2, va="center")
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(COND)), [COND_LABEL[k] for k in COND], fontsize=8)
    ax.set_xlim(-0.5, len(COND) - 0.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("torunun hücre payı (ata ile rekabet)")
    save(fig, "e3_competition")


def fig_variance(v):
    """E5: genom başına ortalama uygunluk ile beden eni geni (g1) ilişkisi."""
    from scipy.stats import norm

    m = v.groupby("genome").agg(n_org=("n_org", "mean"), sd=("n_org", "std"), g1=("g1", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    u1 = norm.cdf(m["g1"])
    ax.errorbar(u1, m["n_org"], yerr=m["sd"], fmt="o", ms=5, color=COND["evrim"], ecolor="#b9cdea", elinewidth=1.2, capsize=0)
    r = np.corrcoef(m["g1"], m["n_org"])[0, 1]
    ax.annotate(f"r = {r:.2f}", (0.97, 0.92), xycoords="axes fraction", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("beden eni geni u1 = Φ(g1)  (dar → geniş)")
    ax.set_ylabel("4000 adımda soydaki birey")
    save(fig, "e5_width_fitness")


MU = {"mutasyonsuz": ("μ = 0", "#eb6834"), "evrim_mu0.05": ("μ = 0,05", "#9bbbe6"), "evrim": ("μ = 0,15", "#5b93dd"),
      "evrim_mu0.5": ("μ = 0,5", "#2a78d6"), "evrim_mu1.0": ("μ = 1,0", "#123f7a")}


def fig_curve(c, name="e5_fitness_curve", conds=None):
    """E5: popülasyon örneklerinin uygunluğu zaman içinde; kesikli çizgi = en iyi kurucu."""
    conds = conds or {k: (COND_LABEL[k], v) for k, v in COND.items()}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for cond, (label, col) in conds.items():
        d = c[c.cond == cond]
        if d.empty:
            continue
        per = d.groupby(["seed", "t"])["n_org"].mean().reset_index()
        g = per.groupby("t")["n_org"].agg(["mean", "std", "count"]).reset_index()
        se = g["std"].fillna(0) / np.sqrt(g["count"])
        ax.fill_between(g["t"], g["mean"] - se, g["mean"] + se, color=col, alpha=0.15, lw=0)
        ax.plot(g["t"], g["mean"], color=col, marker="o", ms=4, label=label)
    f0 = c[c.t == 0].groupby(["cond", "seed", "genome"])["n_org"].mean()
    best = f0.groupby(["cond", "seed"]).max().mean()
    ax.axhline(best, color=INK2, lw=0.8, ls="--")
    ax.annotate("en iyi kurucu", (c.t.max(), best), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("evrim adımı")
    ax.set_ylabel("popülasyon uygunluğu (tek-soy, birey)")
    if len(conds) > 2:
        fig.set_size_inches(5.6, 3.0)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    else:
        ax.legend(fontsize=8, loc="lower right")
    save(fig, name)


def fig_verify(v):
    """E5 doğrulama: kurucular ve her koşulun en iyi son genomları, taze tekrarlarla."""
    order = ["kurucu"] + [k for k in MU if k in set(v.cond)]
    m = v.groupby(["cond", "seed", "genome"]).n_org.mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    rng = np.random.default_rng(0)
    for i, cond in enumerate(order):
        y = m[m.cond == cond].n_org.values
        col = INK2 if cond == "kurucu" else MU[cond][1]
        ax.scatter(i + (rng.random(len(y)) - 0.5) * 0.3, y, s=22, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [y.max()] * 2, color=col, lw=1.5, zorder=4)
    best = m[m.cond == "kurucu"].groupby("seed").n_org.max().mean()
    ax.axhline(best, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(order)), ["kurucular"] + [MU[k][0] for k in order[1:]], fontsize=7)
    ax.set_ylabel("tek-soy uygunluk (birey, 6 taze tekrar)")
    save(fig, "e5_verify")


def main():
    parts = ["# Evrimsel katman — sonuç özeti\n\n`analysis/plots_evo.py` tarafından üretilir.\n"]
    for name, title in (("e1", "E1 – Çeşitlilik ve canlılık"), ("e4", "E4 – Parçadan yenilenme ve yara mutasyonu")):
        p = RES / f"{name}.csv"
        if p.exists():
            parts.append(table(pd.read_csv(p), title))
    p = RES / "e3_history.csv"
    if p.exists():
        fig_history(pd.read_csv(p))
    p = RES / "e3_competition.csv"
    if p.exists():
        c = pd.read_csv(p)
        fig_competition(c)
        g = c.groupby("cond")[["evolved_cell_frac", "genome_distance", "mean_generation"]].agg(["mean", "std", "count"]).round(3)
        g.columns = ["_".join(k) for k in g.columns]
        parts.append(table(g.reset_index(), "E3 – Torun ile kendi atasının rekabeti (0,5 = fark yok)"))
    p = RES / "e5_variance.csv"
    if p.exists():
        fig_variance(pd.read_csv(p))
        parts.append(table(pd.read_csv(RES / "e5_summary.csv"), "E5 – Ata genomları arasında kalıtsal uygunluk farkı (H², tek yönlü ANOVA)"))
    p = RES / "e5_curve.csv"
    if p.exists():
        c = pd.read_csv(p)
        fig_curve(c)
        g = c.groupby(["cond", "t"])["n_org"].agg(["mean", "std", "count"]).round(2).reset_index()
        parts.append(table(g, "E5 – Popülasyon uygunluğu zaman içinde (tek-soy testi, birey)"))
        p2 = RES / "e5_curve_mu.csv"
        if p2.exists():
            cm = pd.concat([c, pd.read_csv(p2)])
            fig_curve(cm, "e5_fitness_curve_mu", MU)
            per = cm.groupby(["cond", "seed", "t"]).n_org.mean().reset_index()
            g = per.groupby(["cond", "t"]).n_org.mean().unstack("t").round(1).reset_index()
            g.columns = [str(k) for k in g.columns]
            parts.append(table(g, "E5 – μ taraması: popülasyon uygunluğu (2 tohum ortalaması, birey)"))
    p = RES / "e5_verify.csv"
    if p.exists():
        v = pd.read_csv(p)
        fig_verify(v)
        m = v.groupby(["cond", "seed", "genome"]).n_org.agg(["mean", "std"]).reset_index()
        g = m.groupby(["cond", "seed"])["mean"].agg(["max", "mean", "count"]).round(2).reset_index()
        parts.append(table(g, "E5 – Doğrulama: genom başına 6 taze tekrar (kurucular: 12'nin tamamı; koşullar: son anın en iyi 4'ü)"))
    p = RES / "e5_confirm.csv"
    if p.exists():
        v = pd.read_csv(p)
        g = v.groupby(["cond", "seed", "genome"]).n_org.agg(["mean", "std", "count"]).reset_index()
        g["se"] = g["std"] / np.sqrt(g["count"])
        parts.append(table(g.round(2), "E5 – Kesin test: en iyi evrimleşmiş genomlar ve en iyi kurucular, 24 taze tekrar"))
    (ROOT / "analysis" / "summary_evo.md").write_text("\n".join(parts))
    print("yazıldı: analysis/summary_evo.md")


if __name__ == "__main__":
    main()
