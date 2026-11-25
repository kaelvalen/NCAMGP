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


# --- adım fiziği ---
from nca.evo import alive_cells  # noqa: E402
from nca.life import E_CH, M_CH, LifePhysics, life_step  # noqa: E402


def tot(x, R):
    return R.sum() + x[:, M_CH].sum() + x[:, E_CH].sum()


def blob_world(m, n=3, size=24, seed=0, e=1.0, r=1.0):
    """n küçük canlı blok, düzgün kaynak."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, m.channels, size, size)
    for i in range(n):
        y, xx = 4 + 7 * i, 4 + 5 * i
        x[0, M_CH, y : y + 3, xx : xx + 3] = 1.0
        x[0, E_CH, y : y + 3, xx : xx + 3] = e
        x[0, 6:, y : y + 3, xx : xx + 3] = torch.rand(m.n_hidden, 3, 3, generator=gen)
    g = torch.randn(1, m.n_genome, size, size, generator=gen) * (x[:, M_CH : M_CH + 1] > 0)
    return x, g, torch.full((1, 1, size, size), r)


def random_model(seed=0):
    torch.manual_seed(seed)
    m = LifeCA()
    torch.nn.init.normal_(m.w2.weight, std=0.1)
    return m


def test_life_step_conserves_nutrient():
    m = random_model()
    x, g, R = blob_world(m)
    t0 = tot(x, R)
    with torch.no_grad():
        for _ in range(50):
            x, g, R, _, _ = life_step(m, x, g, R, LifePhysics())
    assert alive_cells(x).any()
    assert torch.isclose(tot(x, R), t0, rtol=1e-4)


def test_starving_cell_returns_mass_to_R():
    m = LifeCA()
    x, g, R = blob_world(m, n=1, e=0.0, r=0.0)  # enerji yok, kaynak yok
    t0 = tot(x, R)
    with torch.no_grad():
        x, g, R, _, _ = life_step(m, x, g, R, LifePhysics(), fire_rate=1.0)
    assert alive_cells(x).sum() == 0 and x.abs().sum() == 0
    assert torch.isclose(tot(x, R), t0, rtol=1e-4)


def test_resource_never_negative():
    m = random_model(1)
    x, g, R = blob_world(m, n=3, r=0.01)
    with torch.no_grad():
        for _ in range(100):
            x, g, R, _, _ = life_step(m, x, g, R, LifePhysics())
            assert R.min() >= 0
    assert x[:, M_CH].min() >= 0


def test_life_step_backprop():
    m = random_model()
    x, g, R = blob_world(m)
    for _ in range(5):
        x, g, R, _, _ = life_step(m, x, g, R, LifePhysics())
    x[:, M_CH].sum().backward()
    assert m.w1.weight.grad is not None and torch.isfinite(m.w1.weight.grad).all()
    assert m.w1.weight.grad.abs().sum() > 0


# --- yeme ---
from nca.evo import V_CH  # noqa: E402
from nca.life import eat, foreign_mass  # noqa: E402


def two_bodies(m):
    """Bitişik iki blok: A (etiket 2, sütun 2-5) ve B (etiket 7, sütun 6-9)."""
    x = torch.zeros(1, m.channels, H, W)
    tag = torch.zeros(1, 1, H, W)
    x[0, M_CH, 4:10, 2:10] = 1.0
    x[0, E_CH, 4:10, 2:10] = 0.5
    tag[0, 0, 4:10, 2:6], tag[0, 0, 4:10, 6:10] = 2.0, 7.0
    A, B = (tag == 2.0)[0, 0], (tag == 7.0)[0, 0]
    return x, tag, torch.ones(1, 1, H, W), A, B


def test_foreign_mass_sees_only_other_body():
    x, tag, _, A, B = two_bodies(LifeCA())
    F = foreign_mass(x, tag)[0, 0]
    assert F[6, 5] == 3.0 and F[6, 6] == 3.0 and F[6, 3] == 0 and F[6, 8] == 0


def test_eat_transfers_to_attacker():
    x, tag, R, A, B = two_bodies(LifeCA())
    a = torch.zeros(1, 1, H, W)
    a[0, 0][A] = 1.0
    x2, R2, lost, _ = eat(x, R, tag, a, LifePhysics())
    assert x2[0, M_CH][B].sum() < x[0, M_CH][B].sum()
    assert x2[0, E_CH][A].sum() > x[0, E_CH][A].sum()
    assert lost[0, 0][A].sum() == 0 and lost[0, 0][B].sum() > 0


def test_same_tag_cannot_eat():
    x, tag, R, A, B = two_bodies(LifeCA())
    tag = torch.where(tag > 0, torch.full_like(tag, 2.0), tag)
    x2, R2, lost, _ = eat(x, R, tag, torch.ones(1, 1, H, W), LifePhysics())
    assert lost.sum() == 0 and torch.allclose(x2[:, M_CH], x[:, M_CH])
    # saldırı hedef olmasa da maliyetlidir: enerji yalnız maliyet kadar R'ye geçer
    assert torch.allclose(x[:, E_CH] - x2[:, E_CH], torch.full_like(x[:, E_CH], LifePhysics().c_attack) * (x[:, M_CH] > 0))
    assert torch.isclose(tot(x2, R2), tot(x, R), rtol=1e-5)


def test_mutual_attack_conserves_and_nonnegative():
    x, tag, R, A, B = two_bodies(LifeCA())
    phys = LifePhysics(k_eat=2.0)  # kurbanın tamamı yenebilir
    x2, R2, _, _ = eat(x, R, tag, torch.ones(1, 1, H, W), phys)
    assert x2[:, M_CH].min() >= 0 and x2[:, E_CH].min() >= 0
    assert torch.isclose(tot(x2, R2), tot(x, R), rtol=1e-5)


def test_victim_potential_rises():
    x, tag, R, A, B = two_bodies(LifeCA())
    a = torch.zeros(1, 1, H, W)
    a[0, 0][A] = 1.0
    x2, _, _, _ = eat(x, R, tag, a, LifePhysics())
    V = x2[0, V_CH]
    assert V[6, 6] > 0 and V[6, 8] == 0 and V[A].sum() == 0


# --- ön eğitim kayıpları ---
from nca.train_life import body_descriptors, debris_loss  # noqa: E402


def blocks(*rects, size=24):
    m = LifeCA()
    x = torch.zeros(1, m.channels, size, size)
    for y0, y1, x0, x1 in rects:
        x[0, M_CH, y0:y1, x0:x1] = 1.0
        x[0, 0, y0:y1, x0:x1] = 0.7
        x[0, E_CH, y0:y1, x0:x1] = 0.5
    return x


def test_debris_loss_allows_two_viable_bodies():
    x = blocks((2, 7, 2, 7), (12, 17, 12, 17))
    assert debris_loss(x, 10).item() == 0


def test_debris_loss_penalizes_small_fragment():
    x = blocks((2, 7, 2, 7), (12, 14, 12, 14))
    assert debris_loss(x, 10).item() > 0


def test_descriptors_translation_invariant():
    x = blocks((3, 9, 4, 8), (9, 11, 4, 12))  # L biçimli beden
    xs = torch.roll(x, shifts=(7, 5), dims=(2, 3))
    assert torch.allclose(body_descriptors(x), body_descriptors(xs), atol=1e-5)


def test_train_life_smoke(tmp_path):
    from nca.train_life import train
    from nca.utils import ROOT, load_config

    cfg = load_config(ROOT / "configs" / "life.yaml", [
        "iterations=3", "grid=32", "batch_size=2", "pool_size=8", "steps=[8,8]", "homeo_k=4",
        f"out_dir={tmp_path}", "damage.start_iter=0", "snapshot_every=1", "model.hidden_dim=16"])
    out = train(cfg)
    assert (out / "model.pt").exists() and (out / "log.csv").exists()


def test_eat_gain_matches_victim_loss():
    x, tag, R, A, B = two_bodies(LifeCA())
    a = torch.zeros(1, 1, H, W)
    a[0, 0][A] = 1.0
    x2, R2, lost, gain = eat(x, R, tag, a, LifePhysics())
    eloss = x[0, E_CH][B].sum() - x2[0, E_CH][B].sum()
    assert torch.isclose(gain.sum(), lost.sum() + eloss) and gain[0, 0][B].sum() == 0


def test_life_step_reports_uptake():
    m = LifeCA()
    x, g, R = blob_world(m, n=1)
    _, _, _, _, out = life_step(m, x, g, R, LifePhysics())
    assert out["uptake"].sum() > 0 and out["uptake"].shape == R.shape
