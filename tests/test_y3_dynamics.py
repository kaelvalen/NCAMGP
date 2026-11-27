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


def test_perm_test_exact_two_sided():
    from plots_life import perm_test
    assert abs(perm_test([1, 2, 3], [10, 11, 12]) - 0.1) < 1e-12  # C(6,3) = 20 bölüşümün 2'si en az bu kadar uç
    assert perm_test([1, 2, 3], [1, 2, 3]) == 1.0
