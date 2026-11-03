"""Eğitilmiş ağırlıkları demo için JSON'a çevirir.

Her model için demo/models/<ad>.json yazılır ve tümü demo/models.js içinde
`window.NCA_MODELS` olarak paketlenir (index.html dosyadan, sunucusuz açılabilsin diye).

JSON alanları:
  channels, n_hidden, hidden_dim, perception, filters (algılama filtre adları sırasıyla),
  fire_rate,
  w1: hidden_dim x (channels*K) satır-ana düz dizi; giriş indeksi c*K + k
  b1: hidden_dim
  w2: channels x hidden_dim satır-ana düz dizi
  probe (varsa): greedy_order, head_means, tail_means, r2_hidden_all
  palette: baş, gövde, kuyruk, göz renkleri

Kullanım: python export/export_weights.py [koşu_dizinleri...]
Varsayılan: runs/r0_s0 runs/r1_s0 runs/r2_s0 runs/r3_s0
"""

import json
import sys
from pathlib import Path

import numpy as np

from nca.model import PERCEPTIONS
from nca.utils import ROOT, load_run, load_target

LABELS = {
    "r0": "R0 - hasarsız",
    "r1": "R1 - dairesel",
    "r2": "R2 - kesme",
    "r3": "R3 - karma",
}


def export(run_dir):
    run_dir = Path(run_dir)
    model, cfg = load_run(run_dir, "cpu")
    rounded = lambda a: [float(f"{v:.6g}") for v in np.asarray(a, np.float64).ravel()]  # noqa: E731
    w1 = model.w1.weight.detach()[:, :, 0, 0].numpy()
    w2 = model.w2.weight.detach()[:, :, 0, 0].numpy()
    name = run_dir.name
    regime = name.split("_s")[0]
    d = {
        "name": name,
        "label": LABELS.get(regime, name),
        "channels": model.channels,
        "n_hidden": model.n_hidden,
        "hidden_dim": model.hidden_dim,
        "perception": model.perception,
        "filters": PERCEPTIONS[model.perception],
        "fire_rate": model.fire_rate,
        "w1": rounded(w1),
        "b1": rounded(model.w1.bias.detach().numpy()),
        "w2": rounded(w2),
        "palette": load_target(cfg.get("target"))["palette"].tolist(),
    }
    probe = run_dir / "probe.json"
    if probe.exists():
        p = json.loads(probe.read_text())
        d["probe"] = {k: p[k] for k in ("greedy_order", "head_means", "tail_means", "r2_hidden_all", "greedy_r2")}
    return d


def main():
    runs = sys.argv[1:] or [ROOT / "runs" / f"r{i}_s0" for i in range(4)]
    out = ROOT / "demo" / "models"
    out.mkdir(parents=True, exist_ok=True)
    bundle = []
    for r in runs:
        if not (Path(r) / "model.pt").exists():
            print(f"atla (model yok): {r}")
            continue
        d = export(r)
        (out / f"{d['name']}.json").write_text(json.dumps(d))
        bundle.append(d)
        print(f"dışa aktarıldı: {d['name']} ({len(d['w1']) + len(d['b1']) + len(d['w2'])} parametre)")
    js = "// export/export_weights.py tarafından üretildi\nwindow.NCA_MODELS = " + json.dumps(bundle) + ";\n"
    (ROOT / "demo" / "models.js").write_text(js)
    print(f"yazıldı: demo/models.js ({len(bundle)} model)")


if __name__ == "__main__":
    main()
