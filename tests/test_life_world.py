"""Çok canlılı dünya: bölünme nedeni, sönen kesmeler, besin korunumu, tükenme."""

import torch

from nca.life import E_CH, M_CH, LifeCA, LifePhysics
from nca.life_world import LifeParams, LifeWorld


def nutrient(w):
    return float(w.R.sum() + w.x[:, M_CH].sum() + w.x[:, E_CH].sum())


def empty_world(size=32, **kw):
    torch.manual_seed(0)
    m = LifeCA()
    return LifeWorld(m, LifePhysics(), size=size, n_seeds=0, params=LifeParams(min_viable=10, **kw), seed=0)


def test_self_split_gets_new_tag_and_cause_self():
    w = empty_world()
    w.x[0, M_CH, 10:16, 4:20] = 1.0
    w.x[0, E_CH, 10:16, 4:20] = 1.0
    w.tag[0, 0, 10:16, 4:20] = 3.0
    w.track()
    assert sum(o.alive for o in w.orgs.values()) == 1
    w.x[0, :, 10:16, 11:13] = 0  # beden ortadan kendiliğinden ayrıldı (kesme ve saldırı yok)
    w.t += w.p.track_every
    w.retag_splits()
    w.track()
    alive = w.x[0, M_CH] > 0.1
    assert len(torch.unique(w.tag[0, 0][alive])) == 2
    assert any(e[1] == "bolunme" and e[4] == "self" for e in w.events)
    assert w.history[-1]["fission_self"] == 1


def test_cut_rate_anneals_to_zero():
    w = empty_world(cut_rate=0.002, anneal_steps=1000)
    assert w.current_cut_rate() == 0.002
    w.t = 500
    assert abs(w.current_cut_rate() - 0.001) < 1e-12
    w.t = 1000
    assert w.current_cut_rate() == 0


def test_world_step_conserves_without_cuts():
    torch.manual_seed(0)
    m = LifeCA()
    torch.nn.init.normal_(m.w2.weight, std=0.1)
    w = LifeWorld(m, LifePhysics(), size=48, n_seeds=4, params=LifeParams(cut_rate=0.0, mu=1.0), seed=0)
    n0 = nutrient(w)
    for _ in range(100):
        w.step()
    assert abs(nutrient(w) - n0) / n0 < 1e-4


def test_cut_conserves_nutrient():
    w = empty_world()
    w.x[0, M_CH, 8:20, 8:20] = 1.0
    w.x[0, E_CH, 8:20, 8:20] = 1.0
    w.tag[0, 0, 8:20, 8:20] = 3.0
    w.track()
    n0 = nutrient(w)
    w.cut(w.labels == 1, 0.3)
    assert abs(nutrient(w) - n0) / n0 < 1e-5


def test_extinction_does_not_crash():
    torch.manual_seed(0)
    w = LifeWorld(LifeCA(), LifePhysics(), size=32, n_seeds=2, params=LifeParams(), seed=0)
    w.x.zero_()
    for _ in range(20):
        w.step()
    assert w.extinct and w.history[-1]["n_org"] == 0
