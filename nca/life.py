"""Yaşayan ekosistem fiziği (docs/yasayan-ekosistem/tasarim.md).

Durum x (B, C, H, W): [R, G, B, m, V, E, gizli_1..gizli_n]
  m: kütle (eski alfa; hücre m > 0,1 iken canlı), V: yara potansiyeli, E: enerji.
  Kural RGB ve gizli kanalları yazar; m, V, E'yi yalnız fizik değiştirir.
Genom g (B, G, H, W): FiLM ile kuralın gizli katmanını değiştirir (sabit rastgele izdüşümler).
Kaynak R (B, 1, H, W) ve yabancı kütle F (B, 1, H, W) kuralın algıladığı dünya alanlarıdır.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .evo import V_CH
from .model import KERNELS, PERCEPTIONS

RGB = slice(0, 3)
M_CH = 3
E_CH = 5
H0 = 6  # ilk gizli kanal

# akış yönleri: indeks 0 = kal, 1..8 = DIRS sırası (dy, dx)
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

STAY_BIAS = 6.0  # başlangıçta kütle yerinde kalır (yoksa beden ilk adımlarda dağılır)
ATTACK_BIAS = -4.0  # ata saldırmaz; saldırıyı evrim bulur


class LifeCA(nn.Module):
    def __init__(self, n_hidden=12, n_genome=32, hidden_dim=128, perception="sobel_lap", fire_rate=0.5,
                 film_scale=1.0, film_seed=0):
        super().__init__()
        self.n_hidden, self.n_genome, self.hidden_dim = n_hidden, n_genome, hidden_dim
        self.perception, self.fire_rate = perception, fire_rate
        self.film_scale, self.film_seed = film_scale, film_seed
        self.channels = H0 + n_hidden
        n_in = self.channels + 2  # + R, F
        names = PERCEPTIONS[perception]
        bank = torch.stack([KERNELS[n] for n in names])
        self.register_buffer("filters", bank.repeat(n_in, 1, 1)[:, None])
        self.n_in = n_in
        self.n_out = 3 + n_hidden + 1 + 9 + 1
        self.w1 = nn.Conv2d(n_in * len(names), hidden_dim, 1)
        self.w2 = nn.Conv2d(hidden_dim, self.n_out, 1, bias=False)
        nn.init.zeros_(self.w2.weight)
        bias = torch.zeros(1, self.n_out, 1, 1)
        bias[:, 3 + n_hidden + 1] = STAY_BIAS
        bias[:, -1] = ATTACK_BIAS
        self.register_buffer("out_bias", bias)
        gen = torch.Generator().manual_seed(film_seed)
        s = 1 / math.sqrt(n_genome)
        self.register_buffer("P_s", (torch.randn(hidden_dim, n_genome, generator=gen) * s)[:, :, None, None])
        self.register_buffer("P_b", (torch.randn(hidden_dim, n_genome, generator=gen) * s)[:, :, None, None])

    def forward(self, x, R, F, g):
        p = F_.conv2d(torch.cat([x, R, F], 1), self.filters, padding=1, groups=self.n_in)
        z = self.w1(p)
        if self.film_scale:
            z = z * (1 + self.film_scale * F_.conv2d(g, self.P_s)) + self.film_scale * F_.conv2d(g, self.P_b)
        y = self.w2(F_.relu(z)) + self.out_bias
        nh = self.n_hidden
        return {
            "drgb": y[:, :3],
            "dh": y[:, 3 : 3 + nh],
            "grow": torch.sigmoid(y[:, 3 + nh : 4 + nh]),
            "flow": torch.softmax(y[:, 4 + nh : 13 + nh], 1),
            "attack": torch.sigmoid(y[:, 13 + nh : 14 + nh]),
        }

    def config(self):
        return dict(n_hidden=self.n_hidden, n_genome=self.n_genome, hidden_dim=self.hidden_dim,
                    perception=self.perception, fire_rate=self.fire_rate,
                    film_scale=self.film_scale, film_seed=self.film_seed)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
