"""Ata canlı uzayı: genomla sürekli değişen organik planarya ailesi.

Genom g (8 boyut, g ~ N(0, I)) normal dağılım CDF'iyle u = Phi(g) in (0, 1) değerlerine çevrilir;
her u bir özelliği sürer. Üretici yalnız atanın vücut planlarını tanımlar: fizik bu aileyi
büyütmeyi öğrenir, ekosistemde mutasyona uğrayan genomlar bu uzayın dışına çıkabilir
(docs/evrim_tasarim.md, "Ata planarya + evrim").

  u0  boy                       u4  pigment rengi (doğal planarya renkleri arasında döngü)
  u1  en                        u5  benek gücü ve ölçeği
  u2  kulakçık / boyun          u6  desen: düz -> boyuna çizgili -> enine bantlı (yumuşak geçiş)
  u3  baş ucu ve göz aralığı    u7  kuyruk çatalı ve farinks boyu

Doku gövde koordinatlarında (eksen t, yanal |l|) sabit bir gürültü dokusundan okunur: genom
değiştikçe desen bedenle birlikte bükülür ve iki yan simetrik kalır. Gözler hedefteki tek çok
koyu yapıdır (pigment tabanı PIGMENT_FLOOR); baş sayımı buna dayanabilir.

Kullanım:
    rgba = render(genome)            # (SIZE, SIZE, 4) premultiplied float32
    python targets/creature_space.py # galeri: targets/creature_space_gallery.png
"""

from math import erf, sqrt
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates, zoom

SIZE = 48
SS = 4
PIGMENT_FLOOR = 0.24
N_GENOME = 8

# doğal planarya pigmentleri (açık ten, kahve, çikolata, kızıl kahve, kül-siyah, zeytin, turuncu-kahve)
PALETTE = np.array(
    [
        (0.72, 0.58, 0.42),
        (0.45, 0.30, 0.18),
        (0.26, 0.17, 0.11),
        (0.50, 0.24, 0.16),
        (0.22, 0.22, 0.24),
        (0.36, 0.38, 0.20),
        (0.66, 0.38, 0.18),
    ]
)


def _noise_texture(seed=11, n=128):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, tot = 1.0, 0.0
    for o in range(5):
        k = 4 * 2**o
        g = rng.random((k + 3, k + 3))
        out += amp * zoom(g, n / k, order=3)[:n, :n]
        tot += amp
        amp *= 0.55
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


NOISE = _noise_texture()


def phi(g):
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in np.asarray(g, float)])


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, w):
    return a * (1 - w[..., None]) + np.asarray(b) * w[..., None]


def palette_colour(u):
    """Doğal paletin üzerinde döngüsel yumuşak interpolasyon."""
    x = u * len(PALETTE)
    i = int(np.floor(x)) % len(PALETTE)
    f = x - np.floor(x)
    f = f * f * (3 - 2 * f)
    return PALETTE[i] * (1 - f) + PALETTE[(i + 1) % len(PALETTE)] * f


def sample_noise(t, lat, sx, sy, ox=0.0):
    return map_coordinates(NOISE, [(t * sy + ox) % 1 * 127, (np.abs(lat) * sx) % 1 * 127], order=1, mode="wrap")


def render(genome, size=SIZE, ss=SS):
    u = phi(genome)
    n = size * ss
    c = (np.arange(n) + 0.5) / ss
    yy, xx = np.meshgrid(c, c, indexing="ij")

    length = 28 + 16 * u[0]
    wscale = 0.62 + 0.7 * u[1]
    aur = u[2]
    tip = u[3]
    y0 = (size - length) / 2
    cx = size / 2

    # yarı genişlik profili: uç, kulakçık, boyun, gövde, kuyruk
    prof = np.array(
        [
            (0.00, 0.6 + 1.6 * tip),
            (0.035, 2.4 + 1.6 * tip),
            (0.08, 4.2 + 3.6 * aur),
            (0.12, 4.2 + 2.2 * aur),
            (0.18, 5.6 - 1.6 * aur),
            (0.32, 6.8),
            (0.50, 7.2),
            (0.68, 6.6),
            (0.84, 4.3),
            (0.95, 1.9 + 1.2 * u[7]),
            (1.00, 0.4 + 1.6 * u[7]),
        ]
    )
    hwf = PchipInterpolator(prof[:, 0], prof[:, 1])
    t = (yy - y0) / length
    tc = np.clip(t, 0, 1)
    mid = cx + 0.7 * np.sin(np.pi * (1.1 * tc + 0.15)) - 0.7 * np.sin(np.pi * 0.15)
    hw = hwf(tc) * wscale * (1 + 0.03 * np.sin(2 * np.pi * (3.1 * tc + 0.3)))
    lat = (xx - mid) / np.maximum(hw, 1e-3)
    inside = (t >= 0) & (t <= 1) & (np.abs(lat) < 1)
    # çatal kuyruk: kuyruk ucunda orta hatta çentik
    fork = smoothstep(0.55, 1.0, u[7])
    notch = (tc > 1 - 0.14 * fork) & (np.abs(lat) < 0.45 * fork)
    inside &= ~notch
    edge_dist = (1 - np.abs(lat)) * hw

    # pigment
    base = palette_colour(u[4])
    speck_amt = 0.15 + 0.65 * u[5]
    sx, sy = 1.2 + 2.0 * u[5], 2.5 + 4.0 * u[5]
    speck = sample_noise(tc, lat, sx, sy)
    fine = sample_noise(tc, lat, 4.0, 9.0, 0.37)
    rgb = np.broadcast_to(base, (n, n, 3)).copy()
    rgb *= (1 - speck_amt / 2 + speck_amt * speck)[..., None]
    rgb *= (0.9 + 0.2 * fine)[..., None]
    dark = base * 0.45
    light = np.clip(base * 1.5 + 0.12, 0, 0.95)

    # desen: düz (orta hat), boyuna çizgiler, enine bantlar; yumuşak ağırlıklar
    wp = np.exp(-(((u[6] - np.array([0.1, 0.5, 0.9])) / 0.18) ** 2))
    wp /= wp.sum()
    midline = np.exp(-((lat / 0.22) ** 2))
    stripes = np.exp(-(((np.abs(lat) - 0.5) / 0.14) ** 2)) * smoothstep(0.1, 0.2, tc) * (1 - smoothstep(0.88, 0.97, tc))
    bands = (0.5 + 0.5 * np.cos(2 * np.pi * (tc * (5 + 4 * u[5]) + 0.25))) ** 3 * smoothstep(0.18, 0.26, tc)
    rgb = mix(rgb, dark, 0.6 * (wp[0] * midline + wp[1] * stripes + wp[2] * bands * (1 - 0.6 * np.abs(lat))))
    rgb = mix(rgb, light, smoothstep(0.62, 1.0, np.abs(lat)) * 0.7)  # açık kenar

    # bağırsak gölgesi ve farinks
    gut = np.exp(-((lat / 0.11) ** 2)) * smoothstep(0.17, 0.24, tc) * (1 - smoothstep(0.40, 0.46, tc))
    for side in (-1, 1):
        gut = np.maximum(gut, np.exp(-(((lat - side * 0.4) / 0.12) ** 2)) * smoothstep(0.42, 0.5, tc) * (1 - smoothstep(0.84, 0.92, tc)))
    rgb = mix(rgb, light * 0.85, 0.45 * gut)
    ph_len = 0.07 + 0.08 * u[7]
    ph = ((tc - 0.56) / ph_len) ** 2 + (lat / 0.26) ** 2
    rgb = mix(rgb, np.clip(light * 1.1, 0, 0.95), 0.8 * (1 - smoothstep(0.75, 1.0, ph)))

    # pigment tabanı (gözlerden açık kalsın)
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    rgb *= np.maximum(1.0, PIGMENT_FLOOR / np.maximum(luma, 1e-3))[..., None]

    # gözler: beyaz halka + koyu göz kabı
    et = 0.085 + 0.02 * tip
    ex = 1.6 + 1.8 * (1 - tip) * wscale
    for side in (-1, 1):
        ey = y0 + et * length
        ecx = np.interp(ey, c, mid[:, 0]) + side * ex
        d = np.hypot(yy - ey, xx - ecx)
        rgb = mix(rgb, (0.93, 0.90, 0.84), 1 - smoothstep(1.3, 1.7, d))
        dp = np.hypot(yy - (ey - 0.15), xx - (ecx - 0.25 * side))
        rgb = mix(rgb, (0.03, 0.025, 0.03), 1 - smoothstep(0.95, 1.3, dp))

    alpha = inside * (0.6 + 0.4 * smoothstep(0.0, 1.4, edge_dist))
    rgbp = rgb * alpha[..., None]

    def pool(a):
        return a.reshape(size, ss, size, ss, *a.shape[2:]).mean(axis=(1, 3))

    return np.concatenate([pool(rgbp), pool(alpha)[..., None]], -1).astype(np.float32)


def bank(n, seed=0):
    """n rastgele genom ve hedefleri (premultiplied RGBA, (n, 4, SIZE, SIZE))."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, N_GENOME)).astype(np.float32)
    t = np.stack([render(x).transpose(2, 0, 1) for x in g])
    return g, t


def to_image(rgba, bg=(0.96, 0.95, 0.91)):
    a = rgba[..., 3:4]
    return np.clip(np.asarray(bg) * (1 - a) + rgba[..., :3], 0, 1)


def main():
    out = Path(__file__).parent
    g, t = bank(32, seed=1)
    tiles = [to_image(x.transpose(1, 2, 0)) for x in t]
    rows = [np.concatenate(tiles[i : i + 8], 1) for i in range(0, 32, 8)]
    img = (np.concatenate(rows, 0) * 255).round().astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.BICUBIC).save(out / "creature_space_gallery.png")
    print("yazıldı: targets/creature_space_gallery.png")


if __name__ == "__main__":
    main()
