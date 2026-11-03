"""Hasar ve müdahale işlemleri: dairesel silme, yarım düzlem kesmesi, orta parça,
parçaları ayırma ve gizli kanal müdahalesi. Tümü (B, C, H, W) tensörleri üzerinde çalışır."""

import math

import torch

from .model import ALIVE_THRESHOLD


def _grid(h, w, device):
    """Piksel merkezlerinin ızgara merkezine göre koordinatları (y, x)."""
    ys = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2
    xs = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2
    return torch.meshgrid(ys, xs, indexing="ij")


def circle_mask(h, w, center, radius, device="cpu"):
    """center ve radius [-1, 1] normalize koordinatlarında (Mordvintsev, 2020)."""
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) < radius**2


def circle_damage(x, gen=None, r_range=(0.1, 0.4), c_range=(-0.5, 0.5), on_alive=False):
    """Her örnekte rastgele dairesel bölgeyi siler. Silinen maskeyi döndürür.
    on_alive=True ise merkez rastgele bir canlı hücreden seçilir (değerlendirmede hasarın
    organizmayı ıskalamaması için; eğitimde Mordvintsev'deki gibi düzgün dağılım)."""
    b, _, h, w = x.shape
    masks = []
    for i in range(b):
        alive = torch.nonzero(x[i, 3] > ALIVE_THRESHOLD)
        if on_alive and len(alive):
            y, xx = alive[int(torch.randint(len(alive), (1,), generator=gen))].tolist()
            c = [2 * y / (h - 1) - 1, 2 * xx / (w - 1) - 1]
        else:
            c = (torch.rand(2, generator=gen) * (c_range[1] - c_range[0]) + c_range[0]).tolist()
        r = float(torch.rand(1, generator=gen) * (r_range[1] - r_range[0]) + r_range[0])
        m = circle_mask(h, w, c, r, x.device)
        x[i, :, m] = 0
        masks.append(m)
    return torch.stack(masks)


def projection(h, w, angle, device="cpu"):
    """Her pikselin (cos a, sin a) normali üzerindeki izdüşümü."""
    yy, xx = _grid(h, w, device)
    return xx * math.cos(angle) + yy * math.sin(angle)


def cut_threshold(alive, proj, keep_frac):
    """Canlı hücrelerin keep_frac kadarının proj >= eşik tarafında kalacağı eşik."""
    vals = proj[alive]
    if vals.numel() == 0:
        return None
    return torch.quantile(vals, 1.0 - keep_frac).item()


def halfplane_cut(x, angle, keep_frac):
    """Tek örnekte (C, H, W) bir doğru boyunca kesme: canlı kütlenin keep_frac kadarı kalır.
    Kalan taraf normalin gösterdiği taraftır. Tutulan maskeyi döndürür (H, W)."""
    _, h, w = x.shape
    proj = projection(h, w, angle, x.device)
    alive = x[3] > ALIVE_THRESHOLD
    thr = cut_threshold(alive, proj, keep_frac)
    if thr is None:
        return torch.ones(h, w, dtype=torch.bool, device=x.device)
    keep = proj >= thr
    x[:, ~keep] = 0
    return keep


def random_cut(x, gen=None, keep_range=(0.2, 0.8)):
    """Batch'teki her örneğe rastgele açılı yarım düzlem kesmesi uygular."""
    fracs, angles = [], []
    for i in range(x.shape[0]):
        a = float(torch.rand(1, generator=gen)) * 2 * math.pi
        f = float(torch.rand(1, generator=gen)) * (keep_range[1] - keep_range[0]) + keep_range[0]
        halfplane_cut(x[i], a, f)
        fracs.append(f)
        angles.append(a)
    return angles, fracs


def band_cut(x, angle, lo_frac, hi_frac):
    """Canlı kütlenin proj yüzdelikleri [lo_frac, hi_frac] arasındaki şeridi tutar (orta parça)."""
    _, h, w = x.shape
    proj = projection(h, w, angle, x.device)
    alive = x[3] > ALIVE_THRESHOLD
    vals = proj[alive]
    if vals.numel() == 0:
        return torch.ones(h, w, dtype=torch.bool, device=x.device)
    lo = torch.quantile(vals, lo_frac).item()
    hi = torch.quantile(vals, hi_frac).item()
    keep = (proj >= lo) & (proj <= hi)
    x[:, ~keep] = 0
    return keep


def shift(x, dy, dx):
    """Sarmadan (sıfır doldurarak) tam sayı kaydırma. x: (C, H, W)."""
    out = torch.zeros_like(x)
    _, h, w = x.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[:, yd, xd] = x[:, ys, xs]
    return out


def split_and_separate(x, angle, keep_frac, gap):
    """Organizmayı doğru boyunca iki parçaya ayırır ve parçaları normal yönünde
    toplam `gap` piksel uzaklaştırır. x: (C, H, W). Yeni durumu döndürür."""
    _, h, w = x.shape
    proj = projection(h, w, angle, x.device)
    alive = x[3] > ALIVE_THRESHOLD
    thr = cut_threshold(alive, proj, keep_frac)
    if thr is None:
        return x.clone()
    side = proj >= thr
    ny, nx = math.sin(angle), math.cos(angle)
    d = gap / 2
    a = shift(x * side, round(ny * d), round(nx * d))
    b = shift(x * ~side, round(-ny * d), round(-nx * d))
    return a + b


def set_channels(x, mask, channels, values, strength=1.0):
    """Müdahale: maskedeki canlı hücrelerde seçilen kanalları hedef değerlere doğru çeker.
    x: (C, H, W), mask: (H, W) bool, values: kanal başına hedef değer."""
    alive = (x[3] > ALIVE_THRESHOLD) & mask
    for c, v in zip(channels, values):
        cur = x[c][alive]
        x[c][alive] = (1 - strength) * cur + strength * v
    return alive


def mirror_channels(x, channels, axis_lo, axis_hi):
    """Gradyanı ters çevirme müdahalesi: seçilen kanalları [axis_lo, axis_hi] satır aralığında
    dikey eksende aynalar (baş-kuyruk ekseni dikeydir)."""
    seg = x[channels, axis_lo : axis_hi + 1].clone()
    x[channels, axis_lo : axis_hi + 1] = torch.flip(seg, dims=[1])
    return x
