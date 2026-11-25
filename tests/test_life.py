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


# --- taşınım ---
from nca.life import DIRS, transport  # noqa: E402


def one_dir_flow(shape, k):
    """Tüm kütle k yönüne (0 = kal) akar."""
    f = torch.zeros(shape[0], 9, *shape[2:])
    f[:, k] = 1
    return f


def test_transport_conserves_mass_and_energy():
    torch.manual_seed(0)
    m = torch.rand(1, 1, H, W)
    flow = torch.softmax(torch.randn(1, 9, H, W), 1)
    ext = torch.rand(1, 2, H, W)
    m2, ext2, _, _ = transport(m, flow, ext, torch.rand(1, 3, H, W))
    assert torch.allclose(m2.sum(), m.sum(), rtol=1e-5)
    assert torch.allclose(ext2.sum(), ext.sum(), rtol=1e-5)


def test_transport_edge_mass_stays():
    m = torch.zeros(1, 1, H, W)
    m[0, 0, 0, 0] = 1.0
    flow = one_dir_flow(m.shape, 1 + DIRS.index((-1, -1)))
    m2, _, _, _ = transport(m, flow, torch.zeros(1, 1, H, W), torch.zeros(1, 1, H, W))
    assert m2[0, 0, 0, 0] == 1.0 and m2.sum() == 1.0


def test_transport_intensive_is_mass_weighted():
    m = torch.zeros(1, 1, H, W)
    q = torch.zeros(1, 1, H, W)
    m[0, 0, 5, 4], m[0, 0, 5, 6] = 1.0, 3.0
    q[0, 0, 5, 4], q[0, 0, 5, 6] = 0.0, 4.0
    flow = torch.zeros(1, 9, H, W)
    flow[0, 1 + DIRS.index((0, 1)), 5, 4] = 1  # sağa
    flow[0, 1 + DIRS.index((0, -1)), 5, 6] = 1  # sola
    flow[0, 0] += (flow.sum(1)[0] == 0).float()
    m2, _, q2, _ = transport(m, flow, torch.zeros(1, 1, H, W), q)
    assert torch.isclose(m2[0, 0, 5, 5], torch.tensor(4.0))
    assert torch.isclose(q2[0, 0, 5, 5], torch.tensor(3.0))


def test_transport_no_nan_on_empty_grid():
    m = torch.zeros(1, 1, H, W)
    flow = torch.softmax(torch.randn(1, 9, H, W), 1)
    _, _, q2, _ = transport(m, flow, torch.zeros(1, 1, H, W), torch.rand(1, 3, H, W))
    assert torch.isfinite(q2).all() and (q2 == 0).all()


def test_transport_tags_never_mix():
    # iki blok arasında 1 hücrelik boşluk; ikisi de birbirine doğru akar
    m = torch.zeros(1, 1, H, W)
    tag = torch.zeros(1, 1, H, W)
    m[0, 0, 4:10, 2:7], tag[0, 0, 4:10, 2:7] = 1.0, 2.0  # A
    m[0, 0, 4:10, 8:13], tag[0, 0, 4:10, 8:13] = 1.0, 7.0  # B
    flow = torch.zeros(1, 9, H, W)
    flow[0, 1 + DIRS.index((0, 1)), :, :7] = 1
    flow[0, 1 + DIRS.index((0, -1)), :, 7:] = 1
    for _ in range(3):
        m2, _, _, tag2 = transport(m, flow, torch.zeros(1, 1, H, W), torch.zeros(1, 1, H, W), tag)
        for t in (2.0, 7.0):
            assert torch.isclose(m2[tag2 == t].sum(), m[tag == t].sum())
        assert torch.isclose(m2.sum(), m.sum())
        m, tag = m2, tag2
    assert m[0, 0, :, 7].sum() == 0  # boşluğa iki bireyden de kütle girmedi
