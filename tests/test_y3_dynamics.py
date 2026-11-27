"""Y3: eğilimden arındırılmış otokorelasyonun ilk negatif minimumu."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from plots_life import acf_first_min  # noqa: E402


def test_acf_finds_half_period_of_trended_sine():
    t = np.arange(2000)
    lag, val, thr = acf_first_min(np.sin(2 * np.pi * t / 100) + 0.01 * t)  # periyot 100 + doğrusal eğilim
    assert abs(lag - 50) <= 2 and val < -0.8 and thr == 2 / np.sqrt(2000)


def test_acf_white_noise_has_no_significant_minimum():
    _, val, thr = acf_first_min(np.random.default_rng(0).normal(size=2000))
    assert abs(val) < thr * 1.5
