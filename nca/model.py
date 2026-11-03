"""Algılama + güncelleme kuralı (Mordvintsev vd., 2020).

Hücre durumu: [R, G, B, A, gizli_1 ... gizli_n]. Algılama sabit 3x3 filtrelerle yapılır
(kimlik, Sobel-x, Sobel-y, Laplace); güncelleme kuralı tüm hücrelerin paylaştığı
1x1 konvolüsyonlu küçük bir MLP'dir.

Algılama vektörünün düzeni (demo/nca.js ile aynı olmalı): kanal c ve filtre k için
indeks c * K + k; filtre sırası PERCEPTIONS[perception] listesindeki gibidir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ALIVE_THRESHOLD = 0.1

_IDENTITY = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
_SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
_LAPLACE = [[1, 2, 1], [2, -12, 2], [1, 2, 1]]

KERNELS = {
    "identity": torch.tensor(_IDENTITY, dtype=torch.float32),
    "sobel_x": torch.tensor(_SOBEL_X, dtype=torch.float32) / 8.0,
    "sobel_y": torch.tensor(_SOBEL_X, dtype=torch.float32).T / 8.0,
    "laplace": torch.tensor(_LAPLACE, dtype=torch.float32) / 16.0,
}

PERCEPTIONS = {
    "sobel": ["identity", "sobel_x", "sobel_y"],
    "sobel_lap": ["identity", "sobel_x", "sobel_y", "laplace"],
    "lap": ["identity", "laplace"],
}


def alive_mask(x):
    """Kendisi veya 3x3 komşularından biri alfa > 0.1 olan hücreler."""
    return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > ALIVE_THRESHOLD


class CAModel(nn.Module):
    def __init__(self, n_hidden=12, hidden_dim=128, perception="sobel_lap", fire_rate=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.channels = 4 + n_hidden
        self.hidden_dim = hidden_dim
        self.perception = perception
        self.fire_rate = fire_rate

        names = PERCEPTIONS[perception]
        bank = torch.stack([KERNELS[n] for n in names])  # (K, 3, 3)
        k = len(names)
        # depthwise konvolüsyon: çıktı kanalı c*K + k  <-  giriş kanalı c, filtre k
        self.register_buffer("filters", bank.repeat(self.channels, 1, 1)[:, None])
        self.n_filters = k

        self.w1 = nn.Conv2d(self.channels * k, hidden_dim, 1)
        self.w2 = nn.Conv2d(hidden_dim, self.channels, 1, bias=False)
        nn.init.zeros_(self.w2.weight)  # başlangıçta kural hiçbir şeyi değiştirmez

    def perceive(self, x):
        return F.conv2d(x, self.filters, padding=1, groups=self.channels)

    def forward(self, x, fire_rate=None, update_mask=None):
        """Tek adım. update_mask verilirse (B,1,H,W) Bernoulli maskesi yerine o kullanılır."""
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        pre_life = alive_mask(x)
        dx = self.w2(F.relu(self.w1(self.perceive(x))))
        if update_mask is None:
            update_mask = (torch.rand_like(x[:, :1]) <= fire_rate).to(x.dtype)
        x = x + dx * update_mask
        post_life = alive_mask(x)
        return x * (pre_life & post_life).to(x.dtype)

    def rollout(self, x, steps, **kw):
        for _ in range(steps):
            x = self(x, **kw)
        return x

    def config(self):
        return dict(
            n_hidden=self.n_hidden,
            hidden_dim=self.hidden_dim,
            perception=self.perception,
            fire_rate=self.fire_rate,
        )

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def make_seed(batch, size, channels, device="cpu"):
    """Ortada tek hücre: alfa ve gizli kanallar 1, RGB 0."""
    if isinstance(size, int):
        size = (size, size)
    h, w = size
    x = torch.zeros(batch, channels, h, w, device=device)
    x[:, 3:, h // 2, w // 2] = 1.0
    return x


def to_rgb(x):
    """Premultiplied RGBA'yı beyaz zemin üzerinde RGB'ye çevirir."""
    rgb, a = x[:, :3], x[:, 3:4].clamp(0, 1)
    return (1.0 - a + rgb).clamp(0, 1)
