"""Deney betiklerinin ortak yardımcıları: koşu bulma, büyütme, IoU izleme."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from .metrics import batch_iou
from .model import make_seed
from .utils import ROOT, device, load_run, load_target, pad_np, pad_to

RESULTS = ROOT / "experiments" / "results"


def find_runs(patterns):
    """runs/ altında model.pt'si olan ve desenlerden birine uyan koşular."""
    out = []
    for pat in patterns:
        for d in sorted((ROOT / "runs").glob(pat)):
            if (d / "model.pt").exists() and d not in out:
                out.append(d)
    return out


def run_info(run_dir):
    """'r3_s1' -> ('r3', 1)."""
    m = re.match(r"(.+)_s(\d+)$", Path(run_dir).name)
    return (m.group(1), int(m.group(2))) if m else (Path(run_dir).name, 0)


class Ctx:
    """Bir koşu için model + ızgara boyutuna yerleştirilmiş hedef."""

    def __init__(self, run_dir, size):
        self.dev = device()
        self.model, self.cfg = load_run(run_dir, self.dev)
        self.name, self.seed = run_info(run_dir)
        self.size = size
        self.tgt = load_target(self.cfg.get("target"))
        self.target = pad_to(self.tgt["rgba"], size).to(self.dev)
        self.target_mask = (self.target[3] > 0.1).cpu().numpy()
        self.target_area = int(self.target_mask.sum())
        self.palette = self.tgt["palette"]
        self.axial = pad_np(self.tgt["axial"], size, fill=-1)
        self.region = pad_np(self.tgt["region"], size, fill=0)

    def seed_batch(self, n):
        return make_seed(n, self.size, self.model.channels, self.dev)

    @torch.no_grad()
    def grow(self, n, steps=300, chunk=128):
        xs = []
        for i in range(0, n, chunk):
            xs.append(self.model.rollout(self.seed_batch(min(chunk, n - i)), steps))
        return torch.cat(xs)

    @torch.no_grad()
    def track(self, x, steps, every=1):
        """steps adım ilerletir, IoU serisini (T, B) ve son durumu döndürür."""
        series = []
        for t in range(1, steps + 1):
            x = self.model(x)
            if t % every == 0:
                series.append(batch_iou(x, self.target))
        return torch.stack(series), x


def to_np(x):
    """(C, H, W) tensör -> (H, W, C) numpy."""
    return x.permute(1, 2, 0).cpu().numpy()


def base_args(desc, default_runs=("r?_s*",)):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--runs", nargs="*", default=list(default_runs), help="runs/ altındaki glob desenleri")
    ap.add_argument("--n", type=int, default=100, help="model başına örnek sayısı")
    ap.add_argument("--eval-seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    return ap


def mean_std(vals):
    vals = np.asarray(vals, float)
    return float(vals.mean()), float(vals.std(ddof=0))
