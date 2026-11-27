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


def test_bud_keeps_lineage_and_is_not_counted():
    """Eşik altı tomurcuk ebeveynli doğar ve izlenir ama sayılmaz; örtüşmesiz kaysa da soyu etiketten bulunur."""
    w = empty_world()
    w.x[0, M_CH, 10:16, 4:20] = 1.0
    w.x[0, E_CH, 10:16, 4:20] = 1.0
    w.tag[0, 0, 10:16, 4:20] = 3.0
    w.track()
    w.x[0, :, 10:16, 15:17] = 0  # 6x3 = 18 hücrelik tomurcuk koptu (< min_org_area 20)
    w.t += w.p.track_every
    w.retag_splits()
    w.track()
    bud = [o for o in w.orgs.values() if o.alive and o.area == 18]
    assert len(bud) == 1 and bud[0].parent == 0
    assert w.history[-1]["n_org"] == 1
    # tomurcuk 10 adımda boyundan fazla kaydı: konumsal örtüşme yok, soy etiketten
    blk = w.x[0, :, 10:16, 17:20].clone(), w.tag[0, 0, 10:16, 17:20].clone()
    w.x[0, :, 10:16, 17:20] = 0
    w.tag[0, 0, 10:16, 17:20] = 0
    w.x[0, :, 20:26, 17:20], w.tag[0, 0, 20:26, 17:20] = blk
    w.t += w.p.track_every
    w.track()
    assert bud[0].alive and bud[0].area == 18


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


def test_moving_seed_protected_during_grace():
    w = empty_world()
    w.plant(torch.zeros(w.model.n_genome), pos=(10, 10))
    tagv = float(w.tag[0, 0, 10, 10])
    w.x[0, :, 10, 12] = w.x[0, :, 10, 10]  # tohum ekildiği hücreden kaydı
    w.tag[0, 0, 10, 12] = tagv
    w.x[0, :, 10, 10] = 0
    w.tag[0, 0, 10, 10] = 0
    n0 = nutrient(w)
    w.t = 10
    w.track()
    assert w.x[0, M_CH, 10, 12] > 0.1  # koruma süresinde silinmedi
    w.t = w.p.seed_grace + 10
    w.track()
    assert w.x[0, M_CH].sum() == 0 and abs(nutrient(w) - n0) / n0 < 1e-5  # süre bitince döküntü, besin R'ye


def test_background_nibbling_is_not_an_attack():
    w = empty_world()
    w.tag[0, 0, 5, 5], w.tag[0, 0, 9, 9] = 3.0, 6.0
    lost = torch.zeros(1, 1, 32, 32)
    m_after = torch.ones(1, 1, 32, 32)
    lost[0, 0, 5, 5] = 0.01  # atanın arka plan kemirmesi (%1)
    lost[0, 0, 9, 9] = 0.30  # gerçek saldırı
    w.t = 100
    w.note_attacks(lost, m_after)
    assert w.last_attacked.get(w.tag_key(6.0)) == 100 and w.tag_key(3.0) not in w.last_attacked


# --- sarmal dünya ---
def torus_world(size=32, **kw):
    torch.manual_seed(0)
    return LifeWorld(LifeCA(), LifePhysics(torus=True), size=size, n_seeds=0,
                     params=LifeParams(min_viable=10, **kw), seed=0)


def test_label_wrap_joins_across_edges_and_corner():
    import numpy as np
    from nca.life_world import label_wrap
    m = np.zeros((16, 16), bool)
    m[5:9, 0:2] = m[5:9, 14:16] = True  # sol-sağ kenardan geçen tek beden
    m[0, 0] = m[15, 15] = True  # köşegen köşe
    m[10:12, 6:8] = True  # ayrı beden
    lab, n = label_wrap(m)
    assert n == 3
    assert lab[5, 0] == lab[5, 15] and lab[0, 0] == lab[15, 15] and lab[10, 6] not in (lab[5, 0], lab[0, 0])


def test_torus_body_across_edge_is_one_organism_with_wrapped_centroid():
    w = torus_world()
    w.x[0, M_CH, 10:16, 0:4] = w.x[0, M_CH, 10:16, 28:32] = 1.0
    w.x[0, E_CH, 10:16, 0:4] = w.x[0, E_CH, 10:16, 28:32] = 1.0
    w.tag[0, 0, 10:16, 0:4] = w.tag[0, 0, 10:16, 28:32] = 3.0
    w.retag_splits()
    w.track()
    live = [o for o in w.orgs.values() if o.alive]
    assert len(live) == 1 and live[0].area == 48
    cy, cx = live[0].centroid
    assert abs(cy - 12.5) < 1e-6 and min(abs(cx - 31.5), abs(cx + 0.5)) < 1e-6  # kenarın üstünde, ortada değil


def test_torus_cut_across_edge_conserves_and_splits():
    w = torus_world()
    w.x[0, M_CH, 8:20, 0:6] = w.x[0, M_CH, 8:20, 26:32] = 1.0
    w.x[0, E_CH, 8:20, 0:6] = w.x[0, E_CH, 8:20, 26:32] = 1.0
    w.tag[0, 0, 8:20, 0:6] = w.tag[0, 0, 8:20, 26:32] = 3.0
    n0 = nutrient(w)
    comp = (w.x[0, M_CH] > 0.1).cpu().numpy()
    w.cut(comp, 0.0)  # dikey kesme (normal x yönünde), kenar üzerinden
    assert abs(nutrient(w) - n0) / n0 < 1e-5
    alive = w.x[0, M_CH] > 0.1
    assert len(torch.unique(w.tag[0, 0][alive])) == 2


def test_torus_world_step_conserves():
    torch.manual_seed(0)
    m = LifeCA()
    torch.nn.init.normal_(m.w2.weight, std=0.1)
    w = LifeWorld(m, LifePhysics(torus=True), size=48, n_seeds=4,
                  params=LifeParams(cut_rate=0.0, mu=1.0, mu_base=0.1), seed=0)
    n0 = nutrient(w)
    for _ in range(100):
        w.step()
    assert abs(nutrient(w) - n0) / n0 < 1e-4
