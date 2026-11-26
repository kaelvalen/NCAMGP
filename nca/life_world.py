"""Yaşayan ekosistem dünyası (docs/yasayan-ekosistem/tasarim.md, 4 ve 6).

World'ün kimlik etiketlerini, segmentasyonunu, soy takibini ve kesmesini kullanır; adım fiziği
nca/life.py'dendir: kural, taşınım (bireyler kaynaşmaz), yeme, kapalı besin döngüsü. Elle yazılmış
davranış yoktur: hareket, saldırı ve bölünmeye kural karar verir. Dış kesmeler (hava olayı)
anneal_steps boyunca doğrusal olarak sıfıra iner. phys.torus ise dünya sarmaldır: segmentasyon,
merkez, kesme ve taşıma kenardan sarmalanır (ata sarmal arenada eğitildi; kenarlı dünyada sabit
yönlü hareket bireyleri köşeye biriktiriyordu).

Her bölünme nedeniyle kaydedilir: "disturbance" (ebeveyn son cause_window adımda kesildi),
"attack" (ebeveyn bu pencerede yenildi) ya da "self".
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from .evo import mutate
from .life import E_CH, M_CH, eat, foreign_mass, life_step
from .model import ALIVE_THRESHOLD
from .world import World, WorldParams


@dataclass
class LifeParams(WorldParams):
    anneal_steps: int = 50000  # kesme oranı bu adımda sıfıra iner
    cause_window: int = 50  # bölünme nedenini belirlerken geriye bakılan adım
    attack_frac_min: float = 0.05  # hücre kütlesinin bu oranından az kemirme saldırı sayılmaz
    mu_base: float = 0.0  # yaradan bağımsız taban mutasyon (V'ye eklenir; evo.mutate base)


def label_wrap(mask, torus=True):
    """8-bağlantılı bileşen etiketleme; torus ise karşı kenarlar (ve köşeler) komşudur. (lab, n)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not torus or n == 0:
        return lab, n
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for d in (-1, 0, 1):  # satır 0 - satır -1 ve sütun 0 - sütun -1, köşegenler dahil
        for a, b in ((lab[0], np.roll(lab[-1], d)), (lab[:, 0], np.roll(lab[:, -1], d))):
            k = (a > 0) & (b > 0)
            for u, v in set(zip(a[k].tolist(), b[k].tolist())):
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[max(ru, rv)] = min(ru, rv)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq, new = np.unique(roots, return_inverse=True)  # roots[0] = 0 -> 0
    return new[lab].astype(np.int32), len(uniq) - 1


class LifeWorld(World):
    def __init__(self, model, phys, size=192, n_seeds=12, params=None, seed=0, genomes=None, groups=None):
        params = params or LifeParams()
        if genomes is None:  # ortak ata: tüm kurucular g = 0 (çeşitlilik yalnız yara mutasyonundan)
            genomes = torch.zeros(n_seeds, model.n_genome)
        dev = next(model.parameters()).device
        self.extinct = False
        self.eaten_acc = torch.zeros(1, 1, size, size, device=dev)
        self.uptake_acc = torch.zeros(1, 1, size, size, device=dev)
        self.last_attacked = {}  # etiket anahtarı -> son yenildiği adım
        self.last_cut = {}  # organizma kimliği -> son kesildiği adım
        self.org_tag = {}  # canlı organizma -> etiket anahtarı (önceki kayıttan)
        self._ev_mark = 0
        self._prev_cent = {}
        self.seed_tags = []  # (etiket anahtarı, ekildiği adım): koruma süresi
        super().__init__(model, phys, size, n_seeds, params, seed, genomes.to(dev), groups)

    @staticmethod
    def tag_key(t):
        return int(round(float(t) * 1000))

    def nutrient(self):
        return float(self.R.sum() + self.x[:, M_CH].sum() + self.x[:, E_CH].sum())

    # --- tek adım ---
    @torch.no_grad()
    def step(self):
        p = self.p
        F = foreign_mass(self.x, self.tag, self.phys.torus)
        self.x, self.g, self.R, self.tag, out = life_step(self.model, self.x, self.g, self.R, self.phys, tag=self.tag, F=F)
        self.x, self.R, lost, gain = eat(self.x, self.R, self.tag, out["attack"], self.phys)
        self.eaten_acc += self.phys.eta * gain
        self.uptake_acc += out["uptake"]
        self.note_attacks(lost, self.x[:, M_CH : M_CH + 1])
        alive = self.x[:, M_CH : M_CH + 1] > ALIVE_THRESHOLD
        if self.kappa > 0:
            self.g = self.homogenize(self.g, self.tag, alive)
        self.g = mutate(self.g, self.x, p.mu, base=p.mu_base).clamp(-p.g_clip, p.g_clip)
        self.t += 1
        if self.t % p.track_every == 0:
            self.retag_splits()
            self.track()
            self.predation()

    def note_attacks(self, lost, m_after):
        """Hücre kütlesinin en az attack_frac_min'ini kaybettiren yeme saldırı sayılır (atanın küçük
        arka plan saldırı çıktısı teması hafifçe kemirir; bu bölünme nedeni sayılmaz)."""
        hit = lost > self.p.attack_frac_min * (m_after + lost)
        if hit.any():
            for k in np.unique(np.round(self.tag[hit].cpu().numpy() * 1000).astype(np.int64)):
                self.last_attacked[int(k)] = self.t

    def components(self):
        """Etiket başına bağlı bileşenler (canlı hücreler, 8-bağlantı): [(etiket anahtarı, maske, boyut)]."""
        alive = (self.x[0, M_CH] > ALIVE_THRESHOLD).cpu().numpy()
        tags = np.round(self.tag[0, 0].cpu().numpy() * 1000).astype(np.int64) * alive
        out = []
        for tv in np.unique(tags[tags > 0]):
            lab, k = label_wrap(tags == tv, self.phys.torus)
            for i in range(1, k + 1):
                m = lab == i
                out.append((int(tv), m, int(m.sum())))
        return out

    def segment(self, alive):
        """World.segment; sarmal dünyada kenardan geçen beden tek bileşendir."""
        tags = np.round(self.tag[0, 0].cpu().numpy() * 1000).astype(np.int64) * alive
        lab = np.zeros(alive.shape, np.int32)
        n = 0
        for tv in np.unique(tags[tags > 0]):
            l, k = label_wrap(tags == tv, self.phys.torus)
            lab[l > 0] = l[l > 0] + n
            n += k
        return lab, n

    # --- sarmal geometri (World.cut / move kancaları) ---
    def torus_delta(self, d):
        """Sarmal fark: [-size/2, size/2)."""
        s = self.size
        return (d + s / 2) % s - s / 2

    def centroid(self, comp):
        """Merkez; sarmal dünyada dairesel ortalama (kenardan geçen beden kenarda ortalanır)."""
        pts = np.argwhere(comp)
        if not self.phys.torus:
            return tuple(pts.mean(0))
        a = pts * (2 * math.pi / self.size)
        c = np.arctan2(np.sin(a).mean(0), np.cos(a).mean(0)) * self.size / (2 * math.pi)
        return tuple(c % self.size)

    def rel_coords(self, comp):
        if not self.phys.torus:
            return super().rel_coords(comp)
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s]
        cy, cx = self.centroid(comp)
        return self.torus_delta(yy - cy), self.torus_delta(xx - cx)

    def wrap_dst(self, dst):
        return dst % self.size if self.phys.torus else dst

    def retag_splits(self):
        """Aynı etiketin yaşayabilir birden çok bileşeni varsa en büyüğü dışındakiler yeni birey olur."""
        by_tag = {}
        for tv, m, n in self.components():
            if n >= self.p.min_viable:
                by_tag.setdefault(tv, []).append((n, m))
        for tv, comps in by_tag.items():
            comps.sort(key=lambda c: -c[0])
            for _, m in comps[1:]:
                self.tag[0, 0][torch.from_numpy(m).to(self.dev)] = self.new_tag(like=tv / 1000)

    def plant(self, gv, pos=None, tries=50, group=None):
        super().plant(gv, pos, tries, group)
        y, x, t0 = self.planted[-1]
        self.seed_tags.append((self.tag_key(self.tag[0, 0, y, x]), t0))

    def clean_small(self, lab, sizes):
        """World.track'in döküntü temizliği: min_viable'dan küçük bileşenler silinir, besinleri R'ye döner.
        Koruma süresindeki tohumlar etiketleriyle korunur (tohum ekildiği hücreden kayabilir)."""
        grace = {k for k, t0 in self.seed_tags if self.t - t0 < self.p.seed_grace}
        tags = np.round(self.tag[0, 0].cpu().numpy() * 1000).astype(np.int64)
        small = np.zeros(lab.shape, bool)
        for i in np.nonzero(np.asarray(sizes) < self.p.min_viable)[0]:
            m = lab == i + 1
            if int(tags[m][0]) not in grace:
                small |= m
        if small.any():
            sm = torch.from_numpy(small).to(self.dev)
            self.R[0, 0][sm] += self.x[0, M_CH][sm] + self.x[0, E_CH][sm]
            for t in (self.x, self.g, self.tag):
                t[0, :, sm] = 0
            lab[small] = 0

    # --- sönen dış kesmeler ---
    def current_cut_rate(self):
        return self.p.cut_rate * max(0.0, 1 - self.t / self.p.anneal_steps)

    def predation(self):
        p, rate0 = self.p, self.current_cut_rate()
        if rate0 <= 0:
            return
        for l, oid in list(self.label_to_id.items()):
            o = self.orgs[oid]
            rate = min(1.0, rate0 * o.area / p.cut_ref_area)
            if o.area >= p.min_cut_area and self.rng.random() < 1 - (1 - rate) ** p.track_every:
                self.cut(self.labels == l, self.rng.random() * 2 * math.pi)
                self.events.append((self.t, "kesme", oid, o.parent))
                self.last_cut[oid] = self.t

    def cut(self, comp, angle, band=1.0):
        """World.cut; kesilen bant ve ezilen hücrelerin besini dünyaya düzgün dağıtılarak geri verilir
        (kapalı döngü). Miktar küçüktür (bant ~ beden eni) ve 192²'ye yayılınca ihmal edilebilir."""
        before = self.nutrient()
        super().cut(comp, angle, band)
        self.R += (before - self.nutrient()) / self.R.numel()

    # --- izleme ---
    def track(self):
        self._ev_mark = len(self.events)
        self._prev_cent = {oid: o.centroid for oid, o in self.orgs.items() if o.alive}
        super().track()

    def record(self):
        super().record()
        h = self.history[-1]
        live = [o for o in self.orgs.values() if o.alive]
        # bölünme nedenleri: önceki kayıttaki ebeveyn etiketine göre
        counts = {"self": 0, "attack": 0, "disturbance": 0}
        w = self.p.cause_window
        for e in list(self.events[self._ev_mark :]):
            if e[1] != "dogum" or e[3] < 0:
                continue
            par = e[3]
            if self.t - self.last_cut.get(par, -10**9) <= w:
                cause = "disturbance"
            elif self.t - self.last_attacked.get(self.org_tag.get(par, -1), -10**9) <= w:
                cause = "attack"
            else:
                cause = "self"
            counts[cause] += 1
            self.events.append((self.t, "bolunme", e[2], par, cause))
        # canlı organizma -> etiket anahtarı
        tags = np.round(self.tag[0, 0].cpu().numpy() * 1000).astype(np.int64)
        self.org_tag = {}
        for l, oid in self.label_to_id.items():
            cells = tags[self.labels == l]
            if cells.size:
                self.org_tag[oid] = int(cells[0])
        if self.phys.torus:  # World.track düz ortalama yazdı; dairesel merkezle düzelt
            for l, oid in self.label_to_id.items():
                self.orgs[oid].centroid = self.centroid(self.labels == l)
        dist = (lambda a, b: math.hypot(*(self.torus_delta(np.subtract(a, b))))) if self.phys.torus else math.dist
        moves = [dist(o.centroid, self._prev_cent[o.id]) for o in live if o.id in self._prev_cent]
        eaten, up = float(self.eaten_acc.sum()), float(self.uptake_acc.sum())
        self.eaten_acc.zero_()
        self.uptake_acc.zero_()
        h.update({
            "speed": float(np.mean(moves)) / self.p.track_every if moves else 0.0,
            "pred_share": eaten / (eaten + up) if eaten + up > 0 else 0.0,
            "fission_self": counts["self"],
            "fission_attack": counts["attack"],
            "fission_disturbance": counts["disturbance"],
            "cut_rate": self.current_cut_rate(),
            "nutrient_total": self.nutrient(),
        })
        if self.t > 0 and not (self.x[0, M_CH] > ALIVE_THRESHOLD).any():  # hiç canlı madde kalmadı
            self.extinct = True
