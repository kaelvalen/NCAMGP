"""Evrimsel fiziği demo için dışa aktarır: demo/evo_models.js (window.EVO_MODELS).

w1: hidden_dim x (channels*K + n_genome) satır-ana; giriş düzeni [algılama c*K+k ..., genom ...]
w2: channels x hidden_dim satır-ana (V kanalı satırı demoda yok sayılır)

Kullanım: python export/export_evo.py [runs/evo_s0 ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from nca.evo import EvoCA
from nca.model import PERCEPTIONS
from nca.utils import ROOT


def export(run_dir):
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    m = EvoCA(**ck["model_config"])
    m.load_state_dict(ck["state_dict"])
    r = lambda a: [float(f"{v:.6g}") for v in np.asarray(a, np.float64).ravel()]  # noqa: E731
    return {
        "name": Path(run_dir).name,
        "label": Path(run_dir).name,
        "channels": m.channels,
        "n_hidden": m.n_hidden,
        "n_genome": m.n_genome,
        "hidden_dim": m.hidden_dim,
        "perception": m.perception,
        "filters": PERCEPTIONS[m.perception],
        "fire_rate": m.fire_rate,
        "sense_v": m.sense_v,
        "physics": ck["physics"],
        "w1": r(m.w1.weight.detach()[:, :, 0, 0]),
        "b1": r(m.w1.bias.detach()),
        "w2": r(m.w2.weight.detach()[:, :, 0, 0]),
    }


def main():
    runs = sys.argv[1:] or [ROOT / "runs" / "evo_s0"]
    bundle = [export(r) for r in runs if (Path(r) / "model.pt").exists()]
    js = "// export/export_evo.py tarafından üretildi\nwindow.EVO_MODELS = " + json.dumps(bundle) + ";\n"
    (ROOT / "demo" / "evo_models.js").write_text(js)
    print(f"yazıldı: demo/evo_models.js ({len(bundle)} model)")


if __name__ == "__main__":
    main()
