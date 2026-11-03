import math

import numpy as np
import torch

from nca import damage as dmg
from nca.metrics import aligned_iou, batch_iou, classify_outcome, count_heads, count_individuals, first_hit
from nca.model import CAModel, make_seed
from nca.utils import load_target, pad_to


def target_state(size=72, channels=16):
    tgt = load_target()
    x = torch.zeros(channels, size, size)
    x[:4] = pad_to(tgt["rgba"], size)
    x[4:] = x[3:4]
    return x, tgt


def test_zero_init_is_identity():
    m = CAModel()
    assert 9000 < m.n_params() < 11000
    x, _ = target_state()
    y = m(x[None], fire_rate=1.0)
    assert torch.allclose(y[0], x)


def test_seed_layout():
    s = make_seed(2, 72, 16)
    assert s[:, :3].sum() == 0 and s[0, 3:, 36, 36].eq(1).all() and s[0].abs().sum() == 13


def test_halfplane_keep_fraction():
    x, _ = target_state()
    alive0 = (x[3] > 0.1).sum().item()
    for frac in (0.2, 0.5, 0.8):
        for angle in np.linspace(0, 2 * math.pi, 7):
            y = x.clone()
            dmg.halfplane_cut(y, float(angle), frac)
            kept = (y[3] > 0.1).sum().item() / alive0
            assert abs(kept - frac) < 0.06, (frac, angle, kept)


def test_horizontal_cut_keeps_head_side():
    # açı -pi/2: normal yukarıyı gösterir, baş (üst) tarafı kalır
    x, tgt = target_state()
    dmg.halfplane_cut(x, -math.pi / 2, 0.3)
    assert count_heads(x.permute(1, 2, 0).numpy(), tgt["palette"]) == 1
    rows = torch.nonzero(x[3] > 0.1)[:, 0]
    assert rows.max() < 36


def test_split_and_separate_two_components():
    x, _ = target_state(128)
    y = dmg.split_and_separate(x, -math.pi / 2, 0.5, gap=20)
    assert count_individuals((y[3] > 0.1).numpy()) == 2
    assert abs((y[3] > 0.1).sum() - (x[3] > 0.1).sum()) <= 2


def test_metrics_on_target():
    x, tgt = target_state()
    target = pad_to(tgt["rgba"], 72)
    assert batch_iou(x[None], target).item() == 1.0
    m = (x[3] > 0.1).numpy()
    assert aligned_iou(np.roll(m, 5, axis=1), m) == 1.0
    s = x.permute(1, 2, 0).numpy()
    area = (tgt["rgba"][3] > 0.1).sum().item()
    assert classify_outcome(s, tgt["palette"], area) == "tek_bas"


def test_first_hit():
    s = torch.tensor([[0.1, 0.95], [0.92, 0.5], [0.95, 0.99]])
    assert first_hit(s).tolist() == [1, 0]


def test_settle_time():
    from nca.metrics import settle_time

    s = torch.tensor([[0.95, 0.5, 0.5], [0.5, 0.95, 0.95], [0.95, 0.95, 0.5]])
    # örnek 0: son adımda geri döndü (adım 3); 1: adım 2'den beri üstünde; 2: sonda altında
    assert settle_time(s).tolist() == [3, 2, -1]
    assert settle_time(torch.full((4, 1), 0.99)).tolist() == [1]


def test_headfree_band_excludes_head():
    import importlib.util

    spec = importlib.util.spec_from_file_location("d5", "experiments/d5_intervention.py")
    d5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d5)
    x, tgt = target_state()
    s = x.permute(1, 2, 0).numpy()
    lo, hi = d5.headfree_band(s, tgt["palette"])
    band = s.copy()
    band[:lo] = 0
    band[hi + 1 :] = 0
    assert hi > lo and count_heads(band, tgt["palette"]) == 0
