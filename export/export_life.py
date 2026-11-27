"""Yaşayan ekosistemi demo için dışa aktarır: demo/life_models.js (window.LIFE_MODEL), demo/life_states.js
(window.LIFE_STATES). Demo sunucusuz (file://) açıldığı için ikisi de <script> ile yüklenir.

LIFE_MODEL dizileri düz, satır-ana:
  filters: K x 9 (3x3 çekirdekler); algılama girişi n_in = channels + 2 (x, R, F), düzen c*K + k
  w1: hidden_dim x (n_in*K), b1: hidden_dim, w2: n_out x hidden_dim, out_bias: n_out
  film_gamma / film_beta: hidden_dim x n_genome (LifeCA.P_s / P_b); z = z*(1 + s*P_s g) + s*P_b g, s = film_scale
LIFE_STATES[i]: x (H x W x C), g (H x W x G), R (H x W) base64 float16; tag (H x W) base64 float32
  (float16 yakın etiketleri birleştirip kimlik bariyerini bozardı).

Kullanım: uv run python export/export_life.py
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nca.life import LifeCA  # noqa: E402
from nca.utils import ROOT, load_config  # noqa: E402

# (ad, etiket, durum dosyası) — Task 4 durum koşuları
STATES = [("ata", "ata (t = 3000, mutasyon başında)", "runs/life2d_mu1_s0/state_003000.npz"),
          ("mu1", "evrimleşmiş (μ > 0, t = 100k)", "runs/life2d_mu1_s0/state_100000.npz"),
          ("mu0", "mutasyonsuz (μ = 0, t = 100k)", "runs/life2d_mu0_s0/state_100000.npz")]


def r(a):
    return [float(f"{v:.6g}") for v in np.asarray(a, np.float64).ravel()]


def export(run_dir, cfg):
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    m = LifeCA(**ck["model_config"])
    m.load_state_dict(ck["state_dict"])
    K = m.filters.shape[0] // m.n_in
    return {
        "channels": m.channels, "n_hidden": m.n_hidden, "n_genome": m.n_genome, "hidden_dim": m.hidden_dim,
        "n_in": m.n_in, "n_out": m.n_out, "fire_rate": m.fire_rate, "film_scale": cfg["film_scale"],
        "filters": [r(m.filters[k, 0]) for k in range(K)],
        "w1": r(m.w1.weight.detach()[:, :, 0, 0]), "b1": r(m.w1.bias.detach()),
        "w2": r(m.w2.weight.detach()[:, :, 0, 0]), "out_bias": r(m.out_bias),
        "film_gamma": r(m.P_s[:, :, 0, 0]), "film_beta": r(m.P_b[:, :, 0, 0]),
        "physics": ck["physics"], "world": cfg["world"],
    }


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def pack_state(npz_path):
    s = np.load(npz_path)
    x, g = s["x"], s["g"]  # (C, H, W), (G, H, W)
    C, H, W = x.shape
    return {"name": Path(npz_path).parent.name, "label": Path(npz_path).parent.name, "t": int(s["t"]),
            "H": H, "W": W, "C": C, "G": g.shape[0],
            "x": b64(x.transpose(1, 2, 0), np.float16), "g": b64(g.transpose(1, 2, 0), np.float16),
            "R": b64(s["R"], np.float16), "tag": b64(s["tag"], np.float32)}


def main():
    cfg = load_config(ROOT / "configs/life_world.yaml", [])
    model = export(ROOT / "runs" / cfg["run"], cfg)
    states = []
    for name, label, path in STATES:
        st = pack_state(ROOT / path)
        st["name"], st["label"] = name, label
        states.append(st)
    for fname, var, obj in [("life_models.js", "LIFE_MODEL", model), ("life_states.js", "LIFE_STATES", states)]:
        p = ROOT / "demo" / fname
        p.write_text(f"// export/export_life.py tarafından üretildi\nwindow.{var} = " + json.dumps(obj) + ";\n")
        print(f"yazıldı: {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
