"""Ortak bahçe deneyi: genom grupları ve pencere ölçümü."""

import importlib.util
from types import SimpleNamespace

import numpy as np

_spec = importlib.util.spec_from_file_location("y5", "experiments/y5_garden.py")
y5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(y5)


def test_garden_genomes_norms():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(3, 32))
    out = y5.garden_genomes(E, 2, rng)
    groups = [g for g, _, _ in out]
    assert groups.count("ata") == 1 and groups.count("evrim") == 3 and groups.count("rastgele") == 6
    assert np.all(out[0][2] == 0)
    for g, src, v in out:
        if g == "rastgele":
            assert np.isclose(np.linalg.norm(v), np.linalg.norm(E[src]))
            assert v @ E[src] / (np.linalg.norm(v) * np.linalg.norm(E[src])) < 0.9


def test_measure_averages_last_window_and_scales_fission():
    hist = [{"t": t, "n_org": t // 10, "mean_area": 10.0, "cells": 64, "speed": 0.1, "pred_share": 0.0,
             "fission_self": 1, "resource": 0.5} for t in range(10, 101, 10)]
    w = SimpleNamespace(history=hist, t=100, size=16, p=SimpleNamespace(track_every=10))
    m = y5.measure(w, 30)  # t > 70: 80, 90, 100
    assert m["n_org"] == 9 and m["coverage"] == 0.25 and m["R"] == 0.5
    assert m["fission_self"] == 100  # pencere başına 1, 10 adımda bir -> 1000 adımda 100
