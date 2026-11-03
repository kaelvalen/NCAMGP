"""D5 - Müdahale: iki başlı organizma elde edilebilir mi, kalıcı mı? (H4)

Durant vd. (2017) deneyinin NCA karşılığı:
  1. Büyümüş organizmadan orta gövde parçası kesilir (baş ve kuyruk atılır).
  2. Kesmeden hemen sonra kuyruk tarafındaki yara hücrelerinde, D4 probunun konum
     taşıdığını gösterdiği gizli kanallar baş bölgesindeki ortalama değerlere çekilir
     (head_clamp, `hold` adım boyunca) ya da bu kanallar parça boyunca aynalanır (mirror).
     Ağırlıklara dokunulmaz.
  3. 800 adım sonra sonuç sınıflanır (nca.metrics.classify_outcome): tek_bas, iki_bas
     (Janus: iki baş, kuyruk yok), ikiz (yaradan ikinci tam birey; iki baş + kuyruk),
     bassiz, bozulma.
  4. Kalıcılık: iki_bas / ikiz çıkan organizmalar müdahalesiz yeniden kesilir; aynı sınıfı
     koruyanların oranı raporlanır.

Önce D4 çalıştırılmalı (runs/<koşu>/probe.json).
Çıktı: experiments/results/d5.csv, experiments/results/d5/<koşu>_*.png
"""

import csv
import json
import math
from collections import Counter

import numpy as np
import torch

from nca import damage as dmg
from nca.evaluate import RESULTS, Ctx, base_args, find_runs, to_np
from nca.metrics import OUTCOMES, classify_outcome, classify_pixels
from nca.utils import save_grid, set_seed, to_uint8

DOWN = math.pi / 2
CLASSES = OUTCOMES

# (tür, kanal seçimi, güç, tutma adımı)
CONDITIONS = [
    ("none", None, 0, 0),
    ("head_clamp", 1, 1.0, 1),
    ("head_clamp", 3, 1.0, 1),
    ("head_clamp", 3, 1.0, 30),
    ("head_clamp", "all", 1.0, 1),
    ("head_clamp", "all", 1.0, 30),
    ("head_clamp", 3, 0.5, 30),
    ("mirror", 3, 1.0, 1),
    ("mirror", "all", 1.0, 1),
]


def pick_channels(probe, k, n_hidden):
    if k == "all":
        return list(range(4, 4 + n_hidden))
    return [4 + c for c in probe["greedy_order"][:k]]


def trunk_cut(x, lo=0.3, hi=0.7):
    """Orta gövde parçası: canlı kütlenin dikey yüzdelikleri [lo, hi]. Parçanın satır sınırlarını döndürür."""
    keep = dmg.band_cut(x, DOWN, lo, hi)
    rows = torch.nonzero(keep.any(1) & (x[3] > 0.1).any(1))[:, 0]
    return int(rows.min()), int(rows.max())


@torch.no_grad()
def run_condition(ctx, grown, probe, cond, steps, wound):
    kind, k, strength, hold = cond
    model = ctx.model
    x = grown.clone()
    bounds = [trunk_cut(x[i]) for i in range(len(x))]
    if kind != "none":
        chans = pick_channels(probe, k, model.n_hidden)
        vals = [probe["head_means"][c - 4] for c in chans]
    if kind == "mirror":
        for i, (top, bot) in enumerate(bounds):
            dmg.mirror_channels(x[i], chans, top, bot)
    start = x.clone()
    if kind == "head_clamp":
        # kuyruk tarafı yara bölgesi ve altında büyüyen doku (örnek başına satır eşiği)
        rows = torch.arange(x.shape[2], device=x.device)[None, :, None]
        cut_row = torch.tensor([bot - wound + 1 for _, bot in bounds], device=x.device)[:, None, None]
        region = (rows >= cut_row).expand(-1, -1, x.shape[3])
        v = torch.tensor(vals, device=x.device)[None, :, None, None]
    for t in range(steps):
        if kind == "head_clamp" and t < hold:
            m = (region & (x[:, 3] > 0.1))[:, None].float()
            x[:, chans] = x[:, chans] * (1 - strength * m) + strength * m * v
        x = model(x)
    labels = [classify_outcome(to_np(x[i]), ctx.palette, ctx.target_area) for i in range(len(x))]
    return start, x, labels


def headfree_band(state_np, palette, margin=2):
    """Baş/göz dokusu içermeyen en uzun canlı satır aralığının orta %60'ı (satır sınırları)."""
    cls = classify_pixels(state_np, palette)
    alive_rows = (state_np[..., 3] > 0.1).any(1)
    head_rows = ((cls == 0) | (cls == 3)).any(1)
    head_rows = np.convolve(head_rows, np.ones(2 * margin + 1), "same") > 0  # baş çevresinde pay
    ok = alive_rows & ~head_rows
    best, cur = (0, -1), None
    for r, v in enumerate(list(ok) + [False]):
        if v and cur is None:
            cur = r
        elif not v and cur is not None:
            if r - 1 - cur > best[1] - best[0]:
                best = (cur, r - 1)
            cur = None
    lo, hi = best
    span = hi - lo + 1
    return lo + int(0.2 * span), hi - int(0.2 * span)


@torch.no_grad()
def recut(ctx, x, steps):
    """Müdahalesiz yeniden kesme (kalıcılık testi). Parça baş dokusu içermeyen gövde
    bölgesinden alınır; aksi halde parçada zaten bulunan baş 'kalıcılık' gibi görünürdü."""
    x = x.clone()
    for i in range(len(x)):
        lo, hi = headfree_band(to_np(x[i]), ctx.palette)
        keep = torch.zeros(x.shape[2], dtype=torch.bool, device=x.device)
        keep[lo : hi + 1] = True
        x[i][:, ~keep] = 0
    x = ctx.model.rollout(x, steps)
    return x, [classify_outcome(to_np(x[i]), ctx.palette, ctx.target_area) for i in range(len(x))]


def cond_name(cond):
    kind, k, s, h = cond
    return "none" if kind == "none" else f"{kind}_k{k}_s{s}_h{h}"


def main():
    ap = base_args(__doc__, ("r3_s*",))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--wound", type=int, default=3, help="yara bölgesi kalınlığı (piksel)")
    args = ap.parse_args()
    out = RESULTS / "d5"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in find_runs(args.runs):
        probe_path = r / "probe.json"
        if not probe_path.exists():
            print(f"atla (probe.json yok, önce D4): {r.name}")
            continue
        probe = json.loads(probe_path.read_text())
        set_seed(args.eval_seed)
        ctx = Ctx(r, 128)  # ikizler ~2 boy uzar; kenara dayanmasın
        grown = ctx.grow(args.n, 300)
        for cond in CONDITIONS:
            start, end, labels = run_condition(ctx, grown, probe, cond, args.steps, args.wound)
            cnt = Counter(labels)
            persist = {}
            for cls in ("iki_bas", "ikiz"):
                idx = [i for i, lab in enumerate(labels) if lab == cls]
                persist[cls] = float("nan")
                if idx:
                    re_end, re_labels = recut(ctx, end[idx], args.steps)
                    persist[cls] = sum(lab == cls for lab in re_labels) / len(idx)
                    save_grid(out / f"{r.name}_{cond_name(cond)}_recut_{cls}.png", to_uint8(re_end[:8]), scale=2)
            save_grid(
                out / f"{r.name}_{cond_name(cond)}.png",
                [f for pair in zip(to_uint8(start[:6]), to_uint8(end[:6])) for f in pair],
                scale=2,
                ncol=6,
            )
            row = {
                "run": r.name,
                "regime": ctx.name,
                "seed": ctx.seed,
                "condition": cond_name(cond),
                **{c: cnt.get(c, 0) / len(labels) for c in CLASSES},
                "persist_iki_bas": persist["iki_bas"],
                "persist_ikiz": persist["ikiz"],
            }
            rows.append(row)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    path = RESULTS / (args.out or "d5.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: {path}")


if __name__ == "__main__":
    main()
