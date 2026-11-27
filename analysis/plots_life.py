"""Yaşayan ekosistem: koşul başına zaman serileri ve karar noktası özeti.

Girdi: runs/lifeworld_<koşul>_s<tohum>/history.csv, summary.yaml (experiments/y_run.py)
Çıktı: analysis/figures/life_timeseries[_<ad>].png|pdf, analysis/summary_life[_<ad>].md

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
        size = yaml.safe_load((d / "config.yaml").read_text())["size"]
        runs.append({"cond": cond, "seed": int(seed), "h": h, "s": s, "size": size})
    return runs


def fig_timeseries(runs, anneal, suffix=""):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7, 2.0 * len(PANELS)), sharex=True)
    for ax, (col, label) in zip(axes, PANELS):
        for cond, color in COND.items():
            rs = [r["h"].set_index("t")[col].rolling(SMOOTH, min_periods=1).mean() for r in runs if r["cond"] == cond]
            if not rs:
                continue
            df = pd.concat(rs, axis=1)
            mean, se = df.mean(axis=1), df.std(axis=1).fillna(0) / np.sqrt(df.notna().sum(axis=1))
            ax.plot(df.index, mean, color=color, lw=1.4, label=f"{COND_LABEL[cond]} (n = {len(rs)})")
            ax.fill_between(df.index, mean - se, mean + se, color=color, alpha=0.2, lw=0)
        ax.axvline(anneal, color=INK2, ls="--", lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(anneal, axes[0].get_ylim()[1], " dış kesmeler sıfır", color=INK2, fontsize=8, va="top")
    axes[-1].set_xlabel("adım")
    save(fig, "life_timeseries" + suffix)


def window(h, lo, hi):
    w = h[(h["t"] > lo) & (h["t"] <= hi)]
    return {"n_org": w["n_org"].mean(), "speed": w["speed"].mean(), "pred_share": w["pred_share"].mean(),
            "self_per_1k": w["fission_self"].mean()}


def late(r, lo=90000):
    w = r["h"][r["h"]["t"] >= lo]
    return {"n_org": w.n_org.mean(), "area": w.mean_area.mean(), "cov": w.cells.mean() / r["size"] ** 2,
            "R": w.resource.mean(), "age": w.mean_age.mean(), "gen": w.mean_generation.mean(),
            "fis": w.fission_self.mean()}  # load() zaten 1000 adım başına çevirdi


def acf_first_min(x):
    """Doğrusal eğilimi çıkarılmış serinin otokorelasyonunda ilk negatif lobun minimumu: (gecikme, değer, 2/√n)."""
    x = np.asarray(x, float)
    n = len(x)
    t = np.arange(n)
    r = x - np.polyval(np.polyfit(t, x, 1), t)
    acf = np.correlate(r, r, "full")[n - 1 :] / (r @ r)
    neg = np.flatnonzero(acf[: n // 4] < 0)
    if not len(neg):
        return 0, 0.0, 2 / np.sqrt(n)
    end = np.flatnonzero(acf[neg[0] : n // 4] >= 0)  # ilk negatif lob; içindeki gürültü kıpırtıları sayılmaz
    lobe = acf[neg[0] : neg[0] + (end[0] if len(end) else n // 4)]
    k = neg[0] + int(np.argmin(lobe))
    return k, float(acf[k]), 2 / np.sqrt(n)


def y3_lines(runs, lo=50000):
    """Y3: kesmeler söndükten sonra n_org ve pred_share salınımı (tohum başına satır)."""
    lines = ["", f"Y3: t > {lo} için eğilimden arındırılmış otokorelasyonun ilk negatif minimumu (gecikme adım,",
             "değer). |değer| < 2/√n ise salınım yok.\n",
             "| koşul | tohum | birey: gecikme, ACF | avcılık: gecikme, ACF | 2/√n |", "|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["cond"], r["seed"])):
        h = r["h"][r["h"]["t"] > lo]
        dt = r["s"]["params"]["track_every"]
        cells = []
        for col in ("n_org", "pred_share"):
            k, v, thr = acf_first_min(h[col].values)
            cells.append(f"{k * dt}, {v:.2f}" if abs(v) >= thr else f"salınım yok ({v:.2f})")
        lines.append(f"| {r['cond']} | {r['seed']} | {cells[0]} | {cells[1]} | {thr:.3f} |")
    return lines


def fig_price(name, anneal, suffix=""):
    """Tohum başına birikimli Price terimleri (ortak yön u üzerinde): seçilim ve aktarım."""
    f = ROOT / "analysis" / f"price_{name}.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for col, color, label in [("sel_u", "#2a78d6", "seçilim (birey düzeyi)"), ("tr_u", "#eb6834", "aktarım (beden içi)")]:
        for k, (seed, d) in enumerate(df.groupby("seed")):
            ax.plot(d.t, d[col].cumsum(), color=color, lw=1.2, alpha=0.8, label=label if k == 0 else None)
    ax.axvline(anneal, color=INK2, ls="--", lw=0.8)
    ax.set_xlabel("adım")
    ax.set_ylabel("birikimli Δḡ · u")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "life_price" + suffix)


def summary(runs, anneal, suffix=""):
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
    lines += ["", "Geç pencere (t ≥ 90000), koşul başına tohum ortalaması. Bölünme eşik altı tomurcukları da sayar",
              "(track_propagules); kişi başı = bölünme / birey.\n",
              "| koşul | birey | beden (hücre) | kapsama | R | yaş (adım) | kuşak | bölünme / 1000 adım | kişi başı |",
              "|---|---|---|---|---|---|---|---|---|"]
    for cond in COND:
        rs = [r for r in runs if r["cond"] == cond]
        if not rs:
            continue
        m = pd.DataFrame([late(r) for r in rs]).mean()
        lines.append(f"| {cond} | {m.n_org:.0f} | {m.area:.0f} | {m["cov"]:.2f} | {m.R:.3f} | {m.age:.0f} | {m.gen:.1f} | "
                     f"{m.fis:.1f} | {m.fis / m.n_org:.3f} |")
    lines += y3_lines(runs, anneal)
    out = ROOT / "analysis" / f"summary_life{suffix}.md"
    keep = out.read_text().partition("\n## Yorum")[1:] if out.exists() else ("", "")  # elle yazılan yorum korunur
    out.write_text("\n".join(lines) + "\n" + "".join(keep))
    print(f"özet: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lifeworld")
    args = ap.parse_args()
    runs = load(args.name)
    if not runs:
        raise SystemExit(f"koşu yok: runs/{args.name}_mu*_s*")
    anneal = runs[0]["s"]["params"]["anneal_steps"]
    suffix = "" if args.name == "lifeworld" else f"_{args.name}"  # her deney kendi çıktısına
    fig_timeseries(runs, anneal, suffix)
    summary(runs, anneal, suffix)
    fig_price(args.name, anneal, suffix)


if __name__ == "__main__":
    main()
