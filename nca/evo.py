"""Evrimsel NCA: genom koşullu fizik kuralı ve elle yazılmış fizik (kalıtım, yara potansiyeli,
mutasyon, enerji/kaynak). Tasarım: docs/evrim_tasarim.md.

Durum x (B, C, H, W): [R, G, B, A, V, gizli_1..gizli_n]
  V: yara potansiyeli. Kural V'yi algılar ama güncelleyemez; V'nin dinamiği elle yazılmıştır.
Genom g (B, G, H, W): kuralın girdisi; kural değiştiremez. Kalıtım ve mutasyon elle yazılmıştır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ALIVE_THRESHOLD, KERNELS, PERCEPTIONS

V_CH = 4  # potansiyel kanalı


def alive_cells(x):
    """Hücrenin kendisi canlı mı (alfa > 0.1). (B, 1, H, W) bool."""
    return x[:, 3:4] > ALIVE_THRESHOLD


def life_mask(x):
    """Kendisi veya 3x3 komşularından biri canlı."""
    return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > ALIVE_THRESHOLD


def box3(t):
    """3x3 toplam (sıfır dolgulu)."""
    return F.avg_pool2d(t, 3, stride=1, padding=1, count_include_pad=True) * 9


class Physics:
    """Elle yazılmış fizik sabitleri."""

    def __init__(self, v_decay=0.97, v_gain=1.0, v_diffuse=0.2, v_max=2.0, kappa=0.02):
        self.v_decay = v_decay  # potansiyelin adım başına sönümü
        self.v_gain = v_gain  # ölen komşu oranı başına potansiyel artışı
        self.v_diffuse = v_diffuse  # canlı doku içinde yayılma
        self.v_max = v_max
        self.kappa = kappa  # canlı hücrelerde genomun komşulara doğru homojenleşme hızı


class EvoCA(nn.Module):
    def __init__(self, n_hidden=12, n_genome=8, hidden_dim=128, perception="sobel_lap", fire_rate=0.5, sense_v=True):
        super().__init__()
        self.n_hidden, self.n_genome, self.hidden_dim = n_hidden, n_genome, hidden_dim
        self.channels = 5 + n_hidden
        self.perception, self.fire_rate, self.sense_v = perception, fire_rate, sense_v
        names = PERCEPTIONS[perception]
        bank = torch.stack([KERNELS[n] for n in names])
        self.n_filters = len(names)
        self.register_buffer("filters", bank.repeat(self.channels, 1, 1)[:, None])
        self.w1 = nn.Conv2d(self.channels * self.n_filters + n_genome, hidden_dim, 1)
        self.w2 = nn.Conv2d(hidden_dim, self.channels, 1, bias=False)
        nn.init.zeros_(self.w2.weight)
        out_mask = torch.ones(1, self.channels, 1, 1)
        out_mask[:, V_CH] = 0  # kural potansiyeli değiştiremez
        self.register_buffer("out_mask", out_mask)

    def update(self, x, g):
        if not self.sense_v:  # ablasyon: kural potansiyeli göremez
            x = torch.cat([x[:, :V_CH], torch.zeros_like(x[:, V_CH : V_CH + 1]), x[:, V_CH + 1 :]], 1)
        p = F.conv2d(x, self.filters, padding=1, groups=self.channels)
        return self.w2(F.relu(self.w1(torch.cat([p, g], 1)))) * self.out_mask

    def config(self):
        return dict(
            n_hidden=self.n_hidden,
            n_genome=self.n_genome,
            hidden_dim=self.hidden_dim,
            perception=self.perception,
            fire_rate=self.fire_rate,
            sense_v=self.sense_v,
        )

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def step(model, x, g, phys, fire_rate=None):
    """Bir adım: öğrenilen güncelleme + canlılık maskesi + potansiyel + kalıtım.
    Gradyan yalnız x'in RGBA/gizli kanallarından akar; V ve g gradyansızdır."""
    fire_rate = model.fire_rate if fire_rate is None else fire_rate
    was_alive = alive_cells(x)
    pre_life = life_mask(x)
    dx = model.update(x, g)
    m = (torch.rand_like(x[:, :1]) <= fire_rate).to(x.dtype)
    x = x + dx * m
    x = x * (pre_life & life_mask(x)).to(x.dtype)
    with torch.no_grad():
        alive = alive_cells(x)
        af = alive.to(x.dtype)
        died = (was_alive & ~alive).to(x.dtype)
        V = x[:, V_CH : V_CH + 1] * phys.v_decay
        V = V + phys.v_gain * box3(died) / 8
        nb_v, nb_n = box3(V * af), box3(af)
        V = V + phys.v_diffuse * (nb_v / nb_n.clamp(min=1) - V)
        V = (V * af).clamp(0, phys.v_max)
        g = inherit(g, was_alive, alive, phys.kappa)
    x = torch.cat([x[:, :V_CH], V, x[:, V_CH + 1 :]], 1)
    return x, g


def inherit(g, was_alive, alive, kappa):
    """Yeni canlanan hücre canlı komşularının genom ortalamasını alır; canlı hücreler
    komşu ortalamasına doğru yavaşça homojenleşir; ölü hücrenin genomu silinir."""
    wa = was_alive.to(g.dtype)
    cnt = box3(wa)
    mean = box3(g * wa) / cnt.clamp(min=1)
    born = alive & ~was_alive
    g = torch.where(born, mean, g + kappa * (mean - g) * wa)
    return g * alive.to(g.dtype)


def wound(x, removed, phys):
    """Dışarıdan silinen hücrelerin (kesme, yırtıcı) komşularında potansiyel oluşturur.
    removed: (B, 1, H, W) bool, x silindikten sonraki durum."""
    with torch.no_grad():
        alive = alive_cells(x).to(x.dtype)
        V = x[:, V_CH : V_CH + 1] + phys.v_gain * box3(removed.to(x.dtype)) / 8
        x[:, V_CH : V_CH + 1] = (V * alive).clamp(0, phys.v_max)
    return x


def mutate(g, x, mu, gen=None, smooth=2):
    """Potansiyel yüksek olan yerde genom mutasyonu: g += mu * V * xi (uzayca düzgün gürültü)."""
    V = x[:, V_CH : V_CH + 1]
    if mu <= 0 or V.max() <= 0:
        return g
    xi = torch.randn(g.shape, generator=gen, device=g.device) if gen is not None else torch.randn_like(g)
    k = 2 * smooth + 1
    xi = F.avg_pool2d(xi, k, stride=1, padding=smooth) * k  # komşu hücreler benzer mutasyon alır
    return g + mu * V * xi * alive_cells(x).to(g.dtype)


def make_seed(genomes, size, channels):
    """Ortada tek hücre; genomes: (B, G). Döndürür (x, g)."""
    b, gdim = genomes.shape
    x = torch.zeros(b, channels, size, size, device=genomes.device)
    x[:, 3, size // 2, size // 2] = 1.0
    x[:, V_CH + 1 :, size // 2, size // 2] = 1.0
    g = torch.zeros(b, gdim, size, size, device=genomes.device)
    g[:, :, size // 2, size // 2] = genomes
    return x, g
