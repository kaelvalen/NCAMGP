"""Config, hedef yükleme, çalıştırma klasörleri ve görselleştirme yardımcıları."""

import copy
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from .model import CAModel, to_rgb

ROOT = Path(__file__).resolve().parent.parent


def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path, overrides=()):
    """YAML config'i yükler; `base:` anahtarı ile kalıtım ve `a.b=değer` biçiminde geçersiz kılma."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = deep_update(parent, cfg)
    cfg = copy.deepcopy(cfg)
    for o in overrides:
        key, val = o.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_target(path=None):
    """Hedefi ve meta verisini yükler. rgba: (4, 40, 40) premultiplied."""
    path = Path(path) if path else ROOT / "targets" / "planaria.npz"
    if not path.is_absolute():
        path = ROOT / path
    d = np.load(path.with_suffix(".npz"))
    return {
        "rgba": torch.from_numpy(d["rgba"]).permute(2, 0, 1).contiguous(),
        "region": d["region"],
        "axial": d["axial"],
        "lateral": d["lateral"],
        "palette": d["palette"],
    }


def pad_to(t, size):
    """(C, h, w) hedefi (C, size, size) ızgaranın ortasına yerleştirir."""
    c, h, w = t.shape
    out = torch.zeros(c, size, size, dtype=t.dtype)
    y0, x0 = (size - h) // 2, (size - w) // 2
    out[:, y0 : y0 + h, x0 : x0 + w] = t
    return out


def pad_np(a, size, fill=0):
    h, w = a.shape[:2]
    out = np.full((size, size) + a.shape[2:], fill, dtype=a.dtype)
    y0, x0 = (size - h) // 2, (size - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = a
    return out


def load_run(run_dir, map_location=None):
    """Eğitilmiş modeli ve config'ini yükler."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "model.pt", map_location=map_location or device(), weights_only=False)
    model = CAModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(map_location or device()).eval()
    return model, ckpt["config"]


def to_uint8(x):
    """(B, C, H, W) durum -> (B, H, W, 3) uint8."""
    rgb = to_rgb(x.detach().float()).permute(0, 2, 3, 1).cpu().numpy()
    return (rgb * 255).round().astype(np.uint8)


def upscale(img, k):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def save_grid(path, frames, scale=4, ncol=None, pad=2):
    """frames: (N, H, W, 3) uint8 listesi -> tek PNG ızgara."""
    frames = [upscale(f, scale) for f in frames]
    n = len(frames)
    ncol = ncol or n
    nrow = (n + ncol - 1) // ncol
    h, w = frames[0].shape[:2]
    grid = np.full((nrow * (h + pad) + pad, ncol * (w + pad) + pad, 3), 255, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, ncol)
        grid[pad + r * (h + pad) : pad + r * (h + pad) + h, pad + c * (w + pad) : pad + c * (w + pad) + w] = f
    imageio.imwrite(path, grid)


def save_gif(path, frames, scale=4, fps=30):
    frames = [upscale(f, scale) for f in frames]
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
