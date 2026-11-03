"""Evrimsel ekosistem: öğrenilmiş genom koşullu fizik + kaynak/enerji + yırtıcı kesmeleri +
yara potansiyeline bağlı mutasyon + soy takibi. Açık bir fitness yoktur: daha uzun yaşayan ve
parçalandığında daha çok yavru bırakan soylar yayılır (docs/evrim_tasarim.md).

Kullanım (kütüphane):
    w = World(model, phys, size=192, n_seeds=12, params=WorldParams(), seed=0)
    for t in range(20000): w.step()
    w.log  # olay ve popülasyon kayıtları
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .evo import V_CH, alive_cells, box3, inherit, mutate, step, wound


@dataclass
class WorldParams:
    uptake: float = 0.15  # hücre adım başına yerel kaynağın bu oranını alır
    cost: float = 0.006  # canlı hücre başına bakım maliyeti / adım (yerel yenilenmeden biraz fazla: çevreden beslenmek gerekir, kalabalıkta açlık)
    birth_cost: float = 0.3  # yeni hücre komşu enerji ortalamasından bu kadar eksik başlar
    regrow: float = 0.004  # kaynak yenilenme hızı (1 - R ile orantılı)
    r_diffuse: float = 0.25
    e_share: float = 0.5  # enerji canlı doku içinde paylaşılır
    e_share_iters: int = 4  # adım başına paylaşım tekrarı: gövde enerjisi hızla eşitlenir, canlı bütün olarak açlık çeker
    e_max: float = 1.0
    mu: float = 0.15  # mutasyon gücü (V ile çarpılır); 0 = mutasyonsuz kontrol
    g_clip: float = 3.5  # genom sınırı (fizik N(0, I) genomlarla eğitildi)
    min_viable: int = 10  # bundan küçük bileşen yaşayamaz (döküntü)
    seed_grace: int = 150  # yeni ekilmiş tohum bu kadar adım döküntü sayılmaz
    self_tol: float = 0.05  # öz/öz-olmayan: komşu kimlik etiketlerinin yayılımı bunu aşarsa yeni hücre doğamaz (bireyler kaynaşmaz)
    cut_rate: float = 0.002  # ref_area büyüklüğünde organizma için adım başına kesilme olasılığı
    cut_ref_area: float = 300.0  # kesilme olasılığı alanla orantılı: büyük bireyler daha sık bölünür
    min_cut_area: int = 80  # bundan küçük organizma kesilmez
    track_every: int = 10
    min_org_area: int = 20


@dataclass
class Organism:
    id: int
    parent: int
    born: int
    area: int = 0
    genome: list = field(default_factory=list)
    centroid: tuple = (0.0, 0.0)
    max_area: int = 0
    n_children: int = 0
    alive: bool = True
    died: int = -1
    founder: int = -1  # soyun kök atası (başlangıç tohumu)
    first_centroid: tuple = ()  # ilk görüldüğü konum (kurucuyu ekildiği tohuma eşlemek için)
    generation: int = 0


class World:
    def __init__(self, model, phys, size=192, n_seeds=12, params=None, seed=0, genomes=None, groups=None):
        self.model = model
        # kural adımında genom homojenleşmesi kapalı; yerine aynı bireyin hücreleriyle sınırlı
        # homojenleşme uygulanır (değen ayrı bireyler sınırlarından genom sızdırmasın)
        self.kappa = phys.kappa
        self.phys = type(phys)(**{**vars(phys), "kappa": 0.0})
        self.p = params or WorldParams()
        self.size = size
        self.dev = next(model.parameters()).device
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        C, G = model.channels, model.n_genome
        self.x = torch.zeros(1, C, size, size, device=self.dev)
        self.g = torch.zeros(1, G, size, size, device=self.dev)
        self.E = torch.zeros(1, 1, size, size, device=self.dev)
        # kimlik etiketi: kalıtsal ama işlevsiz (kural görmez); kesmede bir parça yeni etiket alır
        self.tag = torch.zeros(1, 1, size, size, device=self.dev)
        self.R = torch.ones(1, 1, size, size, device=self.dev)
        self.t = 0
        self.orgs = {}
        self.next_id = 0
        self.planted = []  # (y, x, t): koruma süresindeki tohumlar
        self.founder_genome = {}  # kök ata kimliği -> ilk genomu
        self.seed_positions = []  # ekilen tohumların sırası ve konumu
        self.labels = np.zeros((size, size), np.int32)
        self.label_to_id = {}
        self.events = []  # (t, tür, id, parent, ...)
        self.history = []  # popülasyon özetleri
        if genomes is None:
            genomes = torch.randn(n_seeds, G, generator=torch.Generator().manual_seed(seed)).to(self.dev)
        for i, gv in enumerate(genomes):
            self.plant(gv, group=None if groups is None else groups[i])
        self.track()

    # --- tohum ---
    def plant(self, gv, pos=None, tries=50, group=None):
        s = self.size
        for _ in range(tries):
            yx = pos or tuple(self.rng.integers(12, s - 12, 2))
            y, x = yx
            if not alive_cells(self.x)[0, 0, max(0, y - 8) : y + 9, max(0, x - 8) : x + 9].any():
                break
        self.x[0, 3, y, x] = 1.0
        self.x[0, V_CH + 1 :, y, x] = 1.0
        self.g[0, :, y, x] = gv
        self.E[0, 0, y, x] = self.p.e_max
        self.tag[0, 0, y, x] = self.new_tag(group=group)
        self.planted.append((int(y), int(x), self.t))
        self.seed_positions.append((int(y), int(x)))

    def homogenize(self, g, tag, alive):
        """Genomu yalnız aynı kimlik etiketli canlı komşuların ortalamasına doğru çeker."""
        k = F.unfold(tag, 3, padding=1)  # (1, 9, HW)
        same = ((k - tag.flatten(2)).abs() < 1e-4) & (F.unfold(alive.float(), 3, padding=1) > 0)
        gu = F.unfold(g, 3, padding=1).view(1, g.shape[1], 9, -1)
        w = same.float().unsqueeze(1)
        mean = (gu * w).sum(2) / w.sum(2).clamp(min=1)
        mean = mean.view_as(g)
        return g + self.kappa * (mean - g) * alive.float()

    TAG_SPLIT = 5.5  # iki grup (ör. torun / ata rekabeti) ayrık etiket aralıklarında: [1, 5.5) ve [5.5, 10)

    def new_tag(self, like=None, group=None):
        """Yeni kimlik etiketi (float32 varyans hesabında hassasiyet için küçük aralık). `like` verilirse
        aynı gruptan seçilir: kesmede doğan parça ebeveyninin grubunda kalır."""
        if like is not None:
            group = int(like >= self.TAG_SPLIT)
        if group is None:
            return float(self.rng.uniform(1, 10))
        return float(self.rng.uniform(1, self.TAG_SPLIT) if group == 0 else self.rng.uniform(self.TAG_SPLIT, 10))

    def tag_group_cells(self):
        """Etiket grubuna göre canlı hücre sayıları (grup 0, grup 1)."""
        t = self.tag[0, 0][self.x[0, 3] > 0.1]
        return int((t < self.TAG_SPLIT).sum()), int((t >= self.TAG_SPLIT).sum())

    # --- tek adım ---
    @torch.no_grad()
    def step(self):
        p = self.p
        was = alive_cells(self.x)
        wf = was.float()
        cnt = box3(wf).clamp(min=1)
        spread = box3(self.tag**2 * wf) / cnt - (box3(self.tag * wf) / cnt) ** 2
        self.x, self.g = step(self.model, self.x, self.g, self.phys)
        alive = alive_cells(self.x)
        foreign = alive & ~was & (spread > p.self_tol**2)  # iki ayrı bireyin arasına doğum yok
        if foreign.any():
            keep = (~foreign).float()
            self.x, self.g = self.x * keep, self.g * keep
            alive = alive & ~foreign
        self.tag = inherit(self.tag, was, alive, 0.0)
        if self.kappa > 0:
            self.g = self.homogenize(self.g, self.tag, alive)
        af = alive.float()
        # enerji: kaynak alımı, bakım, doku içi paylaşım, doğum maliyeti
        take = p.uptake * self.R * af
        self.R = self.R - take
        E = self.E + take - p.cost * af
        nb = box3(E * was.float()) / box3(was.float()).clamp(min=1)
        born = alive & ~was
        E = torch.where(born, nb - p.birth_cost, E)
        for _ in range(p.e_share_iters):
            nb = box3(E * af) / box3(af).clamp(min=1)
            E = E + p.e_share * (nb - E) * af
        E = (E * af).clamp(max=p.e_max)
        starving = alive & (E <= 0)
        if starving.any():  # açlıktan ölen hücreler yara ve potansiyel yaratır
            self.x = self.x * (~starving).float()
            self.g = self.g * (~starving).float()
            self.tag = self.tag * (~starving).float()
            E = E * (~starving).float()
            self.x = wound(self.x, starving, self.phys)
        self.E = E
        # kaynak yenilenmesi ve yayılması
        R = self.R + p.regrow * (1 - self.R)
        self.R = (R + p.r_diffuse * (F.avg_pool2d(R, 3, 1, 1, count_include_pad=False) - R)).clamp(0, 1)
        # yara potansiyeline bağlı mutasyon
        self.g = mutate(self.g, self.x, p.mu).clamp(-p.g_clip, p.g_clip)
        self.t += 1
        if self.t % p.track_every == 0:
            self.track()
            self.predation()

    def segment(self, alive):
        """Bireyler: aynı kimlik etiketini taşıyan bağlı bölgeler (birbirine değen ayrı bireyler
        alfa bağlantısıyla birleşik görünürdü)."""
        tags = np.round(self.tag[0, 0].cpu().numpy() * 1000).astype(np.int64) * alive
        lab = np.zeros(alive.shape, np.int32)
        n = 0
        for tv in np.unique(tags[tags > 0]):
            l, k = ndimage.label(tags == tv, structure=np.ones((3, 3)))
            lab[l > 0] = l[l > 0] + n
            n += k
        return lab, n

    # --- organizma takibi (soy ağacı) ---
    def track(self):
        p = self.p
        a = (self.x[0, 3] > 0.1).cpu().numpy()
        lab, n = self.segment(a)
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else []
        self.planted = [q for q in self.planted if self.t - q[2] < p.seed_grace]
        if n:  # yaşayamayacak kadar küçük döküntüyü temizle (koruma süresindeki tohumlar hariç)
            small_ids = set((1 + np.nonzero(np.asarray(sizes) < p.min_viable)[0]).tolist())
            small_ids -= {int(lab[y, x]) for y, x, _ in self.planted}
            small = np.isin(lab, list(small_ids))
            if small.any():
                m = torch.from_numpy(small).to(self.dev)
                for t in (self.x, self.g, self.E, self.tag):
                    t[0, :, m] = 0
                lab[small] = 0
        prev_lab, prev_map = self.labels, self.label_to_id
        new_map, claimed = {}, {}
        g = self.g[0].cpu().numpy()
        order = np.argsort(-np.asarray(sizes)) if n else []
        for k in order:
            l = k + 1
            if sizes[k] < p.min_org_area:
                continue
            m = lab == l
            ov = prev_lab[m]
            ov = ov[ov > 0]
            parent_id = None
            if ov.size:
                vals, cnt = np.unique(ov, return_counts=True)
                parent_id = prev_map.get(int(vals[np.argmax(cnt)]))
            if parent_id is not None and parent_id not in claimed:
                oid = parent_id  # en büyük parça kimliği korur
                claimed[oid] = True
            else:
                oid = self.next_id
                self.next_id += 1
                par = -1 if parent_id is None else parent_id
                self.orgs[oid] = Organism(oid, par, self.t)
                if par >= 0:
                    self.orgs[par].n_children += 1
                    self.orgs[oid].founder = self.orgs[par].founder
                    self.orgs[oid].generation = self.orgs[par].generation + 1
                else:
                    self.orgs[oid].founder = oid
                self.events.append((self.t, "dogum", oid, par))
            o = self.orgs[oid]
            if not o.genome:
                self.founder_genome.setdefault(oid, g[:, m].mean(1).tolist())
                o.first_centroid = tuple(np.argwhere(m).mean(0))
            o.area = int(sizes[k])
            o.max_area = max(o.max_area, o.area)
            o.genome = g[:, m].mean(1).tolist()
            o.centroid = tuple(np.argwhere(m).mean(0))
            new_map[l] = oid
        live = set(new_map.values())
        for oid, o in self.orgs.items():
            if o.alive and oid not in live:
                o.alive, o.died = False, self.t
                self.events.append((self.t, "olum", oid, o.parent))
        self.labels, self.label_to_id = lab, new_map
        self.record()

    def record(self):
        live = [o for o in self.orgs.values() if o.alive]
        gs = np.array([o.genome for o in live]) if live else np.zeros((0, self.model.n_genome))
        div = float(np.mean(np.linalg.norm(gs[:, None] - gs[None], axis=-1))) if len(gs) > 1 else 0.0
        gm = gs.mean(0) if len(gs) else np.zeros(self.model.n_genome)
        self.history.append(
            {
                "t": self.t,
                "n_org": len(live),
                "cells": int(sum(o.area for o in live)),
                "mean_area": float(np.mean([o.area for o in live])) if live else 0.0,
                "mean_generation": float(np.mean([o.generation for o in live])) if live else 0.0,
                **{f"g{i}": float(v) for i, v in enumerate(gm)},
                "mean_age": float(np.mean([self.t - o.born for o in live])) if live else 0.0,
                "genome_div": div,
                "resource": float(self.R.mean()),
                "V_mean": float(self.x[0, V_CH].mean()),
            }
        )

    # --- yırtıcı: rastgele kesme ve parçaları ayırma ---
    def predation(self):
        p = self.p
        for l, oid in list(self.label_to_id.items()):
            o = self.orgs[oid]
            rate = min(1.0, p.cut_rate * o.area / p.cut_ref_area)
            prob = 1 - (1 - rate) ** p.track_every
            if o.area >= p.min_cut_area and self.rng.random() < prob:
                self.cut(self.labels == l, self.rng.random() * 2 * math.pi)
                self.events.append((self.t, "kesme", oid, o.parent))

    def cut(self, comp, angle, band=1.0):
        """comp (H, W) bool organizma maskesi; centroid'den geçen doğru boyunca keser, bandı siler,
        iki parçayı normal yönünde organizma boyunun yarısı kadar uzaklaştırır. Gidecek yer doluysa
        o hücreler ezilir (kaybolur)."""
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s]
        cy, cx = np.argwhere(comp).mean(0)
        ny, nx = math.sin(angle), math.cos(angle)
        proj = (xx - cx) * nx + (yy - cy) * ny
        cut_band = comp & (np.abs(proj) < band)
        side_a, side_b = comp & (proj >= band), comp & (proj <= -band)
        length = proj[comp].max() - proj[comp].min()
        d = int(length / 2 + 3)
        dev = self.dev
        removed = torch.from_numpy(cut_band).to(dev)[None, None]
        for t in (self.x, self.g, self.E, self.tag):
            t[:, :, removed[0, 0]] = 0
        self.x = wound(self.x, removed, self.phys)
        # B parçası yeni birey: yeni kimlik etiketi
        mb = torch.from_numpy(side_b).to(dev) & (self.tag[0, 0] != 0)
        if mb.any():
            self.tag[0, 0][mb] = self.new_tag(like=float(self.tag[0, 0][mb][0]))
        for side, sign in ((side_a, 1), (side_b, -1)):
            self.move(side, round(sign * ny * d), round(sign * nx * d))

    def move(self, mask, dy, dx):
        s = self.size
        src = np.argwhere(mask)
        if not len(src):
            return
        dst = src + [dy, dx]
        ok = (dst[:, 0] >= 0) & (dst[:, 0] < s) & (dst[:, 1] >= 0) & (dst[:, 1] < s)
        sy, sx = torch.from_numpy(src[:, 0]).to(self.dev), torch.from_numpy(src[:, 1]).to(self.dev)
        fields = (self.x, self.g, self.E, self.tag)
        piece = [t[0, :, sy, sx].clone() for t in fields]
        for t in fields:
            t[0, :, sy, sx] = 0
        occupied = alive_cells(self.x)[0, 0].cpu().numpy()
        ok &= ~occupied[np.clip(dst[:, 0], 0, s - 1), np.clip(dst[:, 1], 0, s - 1)]
        # soy takibi için etiketler de parçayla taşınır (ebeveyn kimliği örtüşmeden bulunur)
        lab = self.labels[src[:, 0], src[:, 1]].copy()
        self.labels[src[:, 0], src[:, 1]] = 0
        self.labels[dst[ok, 0], dst[ok, 1]] = lab[ok]
        ty = torch.from_numpy(dst[ok, 0]).to(self.dev)
        tx = torch.from_numpy(dst[ok, 1]).to(self.dev)
        keep = torch.from_numpy(ok).to(self.dev)
        for t, pc in zip(fields, piece):
            t[0, :, ty, tx] = pc[:, keep]

    def summary(self):
        return {"params": asdict(self.p), "t": self.t, "n_orgs_total": len(self.orgs)}
