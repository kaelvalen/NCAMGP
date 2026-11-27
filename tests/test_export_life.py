"""Yaşayan ekosistem dışa aktarımı: model boyutları ve durum paketinin geri çözülmesi."""

import base64
import importlib.util
from dataclasses import asdict

import numpy as np
import torch

from nca.life import LifeCA, LifePhysics

_spec = importlib.util.spec_from_file_location("export_life", "export/export_life.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def unpack(s, dtype):
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


def test_export_roundtrip(tmp_path):
    m = LifeCA(n_hidden=4, n_genome=8, hidden_dim=16)
    torch.save({"state_dict": m.state_dict(), "model_config": m.config(), "physics": asdict(LifePhysics())},
               tmp_path / "model.pt")
    d = ex.export(tmp_path, {"film_scale": 0.5, "world": {"mu": 0.05}})
    K = len(d["filters"])
    assert len(d["w1"]) == m.hidden_dim * m.n_in * K
    assert len(d["film_gamma"]) == m.n_genome * m.hidden_dim and len(d["film_beta"]) == m.n_genome * m.hidden_dim
    assert len(d["w2"]) == m.n_out * m.hidden_dim and len(d["out_bias"]) == m.n_out
    assert d["film_scale"] == 0.5 and d["world"]["mu"] == 0.05

    rng = np.random.default_rng(0)
    x, g = rng.normal(size=(10, 6, 5)).astype(np.float32), rng.normal(size=(8, 6, 5)).astype(np.float32)
    R, tag = rng.random((6, 5)).astype(np.float32), np.round(rng.random((6, 5)) * 9, 3).astype(np.float32)
    np.savez(tmp_path / "state_000100.npz", x=x, g=g, R=R, tag=tag, t=100)
    s = ex.pack_state(tmp_path / "state_000100.npz")
    assert (s["t"], s["H"], s["W"], s["C"], s["G"]) == (100, 6, 5, 10, 8)
    assert np.allclose(unpack(s["x"], np.float16).reshape(6, 5, 10), x.transpose(1, 2, 0), atol=1e-2)
    assert np.allclose(unpack(s["g"], np.float16).reshape(6, 5, 8), g.transpose(1, 2, 0), atol=1e-2)
    assert np.allclose(unpack(s["R"], np.float16).reshape(6, 5), R, atol=1e-2)
    assert np.array_equal(unpack(s["tag"], np.float32).reshape(6, 5), tag)  # kimlik birebir: float16 etiketleri birleştirir
