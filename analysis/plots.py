"""Deney CSV'lerinden makale şekillerini ve özet tablolarını üretir.

Girdi: experiments/results/d*.csv
Çıktı: analysis/figures/*.png|pdf, analysis/summary.md

Kullanım: python analysis/plots.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "analysis" / "figures"

# Kategorik paletin sabit sırası (rejim kimliği; sıra değil kimlik izler)
SERIES = {"r0": "#2a78d6", "r1": "#eb6834", "r2": "#1baf7a", "r3": "#eda100"}
LABEL = {"r0": "R0 hasarsız", "r1": "R1 dairesel", "r2": "R2 kesme", "r3": "R3 karma"}
OUTCOME = {"tek_bas": "#2a78d6", "iki_bas": "#eb6834", "ikiz": "#1baf7a", "bassiz": "#eda100", "bozulma": "#e87ba4"}
OUTCOME_LABEL = {"tek_bas": "tek baş", "iki_bas": "iki baş (Janus)", "ikiz": "ikiz (yaradan yeni birey)", "bassiz": "başsız", "bozulma": "bozulma"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    }
)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"şekil: analysis/figures/{name}.png")


def curve(ax, df, x, y, regimes, xfmt=lambda v: v):
    """Rejim başına tohumlar üzerinden ortalama +- std bandı (eğriler 1,0'da birleştiği için
    doğrudan etiket yerine lejant)."""
    for reg in regimes:
        d = df[df.regime == reg]
        if d.empty:
            continue
        g = d.groupby(x)[y].agg(["mean", "std"]).reset_index().fillna(0)
        c = SERIES.get(reg, INK2)
        ax.fill_between(g[x], (g["mean"] - g["std"]).clip(0, 1), (g["mean"] + g["std"]).clip(0, 1), color=c, alpha=0.15, lw=0)
        ax.plot(g[x], g["mean"], color=c, marker="o", ms=4, label=LABEL.get(reg, reg))
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: xfmt(v)))


def pct(v):
    return f"%{v * 100:.0f}"


def fig_d2():
    out = []
    p = RES / "d2_inplace.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        curve(ax, df, "keep_frac", "success", list(SERIES), pct)
        ax.set_xlabel("kalan parça oranı (canlı kütle)")
        ax.set_ylabel("yenilenme başarısı (IoU ≥ 0,9, 200 adım)")
        ax.legend(loc="lower right", fontsize=8)
        save(fig, "d2_fragment_curve")
        t = df.groupby(["regime", "keep_frac"])["success"].agg(["mean", "std"]).unstack(0)
        out.append("## D2 – Parça boyutu–yenilenme (yerinde)\n\n" + fmt_table(t))
    p = RES / "d2_split.csv"
    if p.exists():
        df = pd.read_csv(p)
        pieces = df[df.kind == "piece"]
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        curve(ax, pieces, "frac", "piece_regenerated", list(SERIES), pct)
        ax.set_xlabel("parça oranı (ayrılan parçanın canlı kütlesi)")
        ax.set_ylabel("parçadan tam birey (hizalı IoU ≥ 0,9)")
        ax.legend(loc="lower right", fontsize=8)
        save(fig, "d2_split_curve")
        cuts = df[df.kind == "cut"]
        t = cuts.groupby(["regime", "frac"])["both_regenerated"].agg(["mean", "std"]).unstack(0)
        out.append("## D2 – Bölünme: iki parçanın da tam bireye dönüşmesi\n\n" + fmt_table(t))
    return out


def fmt_table(t):
    """mean/std sütunlu çok seviyeli tabloyu 'ort ± std' markdown tablosuna çevirir."""
    regs = sorted({c[1] for c in t.columns})
    lines = ["| " + t.index.name + " | " + " | ".join(LABEL.get(r, r) for r in regs) + " |",
             "|---" * (len(regs) + 1) + "|"]
    for idx, row in t.iterrows():
        cells = [f"{row[('mean', r)]:.2f} ± {row[('std', r)]:.2f}" if not np.isnan(row[("mean", r)]) else "–" for r in regs]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table_d1():
    p = RES / "d1.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    cols = ["iou_200", "iou_1000", "circle_repair_rate", "circle_repair_time", "exploded_1000", "faded_1000"]
    g = df.groupby("regime")[cols].agg(["mean", "std"])
    lines = ["| rejim | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for reg, row in g.iterrows():
        lines.append(f"| {LABEL.get(reg, reg)} | " + " | ".join(f"{row[(c, 'mean')]:.3f} ± {row[(c, 'std')]:.3f}" for c in cols) + " |")
    return ["## D1 – Büyüme, dairesel onarım, kararlılık\n\nHedefler: iou_200 ≥ 0,95; circle_repair_rate > 0,95; 1000 adımda bozulma yok.\n\n" + "\n".join(lines) + "\n"]


def table_d3():
    p = RES / "d3.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    g = df.groupby(["fragment", "keep_frac", "regime"])["success"].agg(["mean", "std"]).unstack("regime")
    regs = sorted(df.regime.unique())
    lines = ["| parça | oran | " + " | ".join(LABEL.get(r, r) for r in regs) + " |", "|---" * (len(regs) + 2) + "|"]
    for (frag, k), row in g.iterrows():
        lines.append(f"| {frag} | {k} | " + " | ".join(f"{row[('mean', r)]:.2f} ± {row[('std', r)]:.2f}" for r in regs) + " |")
    return ["## D3 – Kesim yönü (yerinde yenilenme başarısı)\n\n" + "\n".join(lines) + "\n"]


def fig_d4():
    import json

    p = RES / "d4.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    runs = sorted(ROOT.glob("runs/r3_s*/probe.json"))
    if runs:
        probes = [json.loads(r.read_text()) for r in runs]
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        for k, pr in enumerate(probes):
            ax.plot(range(1, len(pr["greedy_r2"]) + 1), pr["greedy_r2"], color=SERIES["r3"],
                    alpha=0.5 + 0.5 * (k == 0), marker="o", ms=4, label="R3 (tohum başına)" if k == 0 else None)
        ax.axhline(0.8, color=INK2, lw=0.8, ls="--")
        ax.annotate("H3 eşiği R² = 0,8", (len(probes[0]["greedy_r2"]), 0.8), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=INK2)
        ax.set_xlabel("açgözlü seçilen gizli kanal sayısı")
        ax.set_ylabel("eksen konumu prob R² (5 katlı CV)")
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
        ax.legend(loc="lower right", fontsize=8)
        save(fig, "d4_probe_greedy")
    cols = ["r2_hidden_all", "r2_hidden_quad", "r2_rgba", "r2_best_single", "r2_greedy_3"]
    g = df.groupby("regime")[cols].agg(["mean", "std"])
    lines = ["| rejim | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for reg, row in g.iterrows():
        lines.append(f"| {LABEL.get(reg, reg)} | " + " | ".join(f"{row[(c, 'mean')]:.3f} ± {row[(c, 'std')]:.3f}" for c in cols) + " |")
    return ["## D4 – Doğrusal prob (eksen konumu)\n\n" + "\n".join(lines) + "\n"]


def fig_d5():
    p = RES / "d5.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    classes = list(OUTCOME)
    conds = list(dict.fromkeys(df.condition))
    regs = [r for r in SERIES if r in set(df.regime)]
    g = df.groupby(["regime", "condition"])[classes + ["persist_iki_bas", "persist_ikiz"]].mean()
    fig, axes = plt.subplots(1, len(regs), figsize=(2.4 * len(regs) + 1.6, 0.3 * len(conds) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(conds))
    for ax, reg in zip(axes, regs):
        left = np.zeros(len(conds))
        for c in classes:
            v = np.array([g.loc[(reg, k), c] for k in conds])
            ax.barh(y, v, left=left, color=OUTCOME[c], height=0.7, label=OUTCOME_LABEL[c], edgecolor="white", linewidth=1)
            left += v
        ax.set_title(LABEL[reg], fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, conds, fontsize=7.5)
    axes[0].invert_yaxis()
    fig.supxlabel("sonuç sınıfı oranı (orta gövde parçası, 800 adım, 3 tohum)", fontsize=9)
    axes[0].legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.12), fontsize=8)
    save(fig, "d5_outcomes")
    fmt = lambda v: "–" if np.isnan(v) else f"{v:.2f}"  # noqa: E731
    lines = ["| rejim | koşul | " + " | ".join(OUTCOME_LABEL[c] for c in classes) + " | kalıcılık: iki baş | kalıcılık: ikiz |", "|---" * (len(classes) + 4) + "|"]
    for reg in regs:
        for cond in conds:
            row = g.loc[(reg, cond)]
            lines.append(f"| {LABEL[reg]} | {cond} | " + " | ".join(f"{row[c]:.2f}" for c in classes) + f" | {fmt(row['persist_iki_bas'])} | {fmt(row['persist_ikiz'])} |")
    return ["## D5 – Müdahale ve kalıcılık\n\nKalıcılık: ikiz/iki baş çıkan organizmadan baş dokusu içermeyen gövde parçası müdahalesiz yeniden kesilir; aynı sınıfı koruyanların oranı.\n\n" + "\n".join(lines) + "\n"]


def table_simple(name, title):
    p = RES / f"{name}.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    lines = ["| " + " | ".join(df.columns) + " |", "|---" * len(df.columns) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return [f"## {title}\n\n" + "\n".join(lines) + "\n"]


def main():
    parts = ["# Sonuç özeti\n\nBu dosya `analysis/plots.py` tarafından üretilir. Değerler tohumlar üzerinden ortalama ± std.\n"]
    parts += table_d1()
    parts += fig_d2()
    parts += table_d3()
    parts += fig_d4()
    parts += fig_d5()
    parts += table_simple("ablation_d1", "Ablasyonlar – D1 metrikleri")
    parts += table_simple("ablation_d2", "Ablasyonlar – D2 yerinde eğri")
    parts += table_simple("d6", "D6 – Gürültüden başlatma")
    (ROOT / "analysis" / "summary.md").write_text("\n".join(parts))
    print("yazıldı: analysis/summary.md")


if __name__ == "__main__":
    main()
