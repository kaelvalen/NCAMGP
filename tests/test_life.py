"""Yaşayan ekosistem fiziği: FiLM kuralı, taşınım, adım fiziği, yeme, ön eğitim kayıpları."""

import torch

from nca.life import LifeCA

H = W = 16


def rand_state(m, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(1, m.channels, H, W, generator=gen)
    x[:, 3] = torch.rand(1, H, W, generator=gen)
    return x


def inputs(m):
    return rand_state(m), torch.rand(1, 1, H, W), torch.zeros(1, 1, H, W)


def test_film_zero_genome_matches_plain():
    torch.manual_seed(0)
    m = LifeCA()
    torch.nn.init.normal_(m.w2.weight, std=0.1)
    x, R, F = inputs(m)
    g0 = torch.zeros(1, m.n_genome, H, W)
    a = m(x, R, F, g0)
    m.film_scale = 5.0
    b = m(x, R, F, g0)
    assert all(torch.allclose(a[k], b[k]) for k in a)


def test_film_genome_changes_output():
    torch.manual_seed(0)
    m = LifeCA()
    torch.nn.init.normal_(m.w2.weight, std=0.1)
    x, R, F = inputs(m)
    a = m(x, R, F, torch.zeros(1, m.n_genome, H, W))
    b = m(x, R, F, torch.randn(1, m.n_genome, H, W))
    assert max((a[k] - b[k]).abs().max().item() for k in a) > 1e-3


def test_init_is_sedentary_and_peaceful():
    m = LifeCA()
    x, R, F = inputs(m)
    out = m(x, R, F, torch.zeros(1, m.n_genome, H, W))
    assert set(out) == {"drgb", "dh", "grow", "flow", "attack"}
    assert out["flow"].shape[1] == 9 and out["dh"].shape[1] == m.n_hidden
    assert (out["flow"][:, 0] > 0.95).all() and (out["attack"] < 0.02).all()
    assert torch.allclose(out["flow"].sum(1), torch.ones_like(out["flow"][:, 0]))
