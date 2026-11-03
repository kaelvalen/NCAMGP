"""Çekirdek katmanın planarya benzeri 40x40 RGBA hedefini çizer (çizgi film; taban çizgisi).

Organik hedef için: targets/make_target_organic.py.

Baş yukarıda, kuyruk aşağıdadır. Bölgeler ayrı renklerle boyanır (baş, gövde, kuyruk,
göz lekeleri); iki başlı sonucu otomatik saymak bu renklere dayanır. Görüntü 16x alt
örnekleme ile kenar yumuşatılarak üretilir.

Çıktılar:
  targets/planaria.png       premultiplied olmayan RGBA (görüntüleme için)
  targets/planaria.npz       rgba (premultiplied, float32), region (0 arka plan, 1 baş,
                             2 gövde, 3 kuyruk, 4 göz), axial (baş ucunda 0, kuyruk ucunda 1),
                             lateral (orta hatta 0, kenarlarda +-1), palette
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator

SIZE = 40
SS = 16  # alt örnekleme çarpanı
CX = SIZE / 2
Y0, LENGTH = 2.0, 36.0  # baş ucu ve toplam boy (piksel)

HEAD_END, TAIL_START = 0.25, 0.72

PALETTE = {
    "head": (0.95, 0.55, 0.66),  # pembe baş
    "body": (0.58, 0.40, 0.22),  # kahverengi gövde
    "tail": (0.28, 0.20, 0.13),  # koyu kuyruk
    "eye": (0.06, 0.06, 0.10),  # göz lekesi
}

# (eksen konumu t, yarı genişlik piksel): kulakçıklı üçgen baş, boyun, gövde, sivrilen kuyruk
_PROFILE = np.array(
    [
        (0.00, 1.6),
        (0.04, 3.6),
        (0.10, 6.6),
        (0.15, 6.2),
        (0.22, 4.9),
        (0.35, 6.3),
        (0.55, 6.7),
        (0.72, 5.4),
        (0.88, 2.9),
        (1.00, 0.5),
    ]
)
_half_width = PchipInterpolator(_PROFILE[:, 0], _PROFILE[:, 1])

EYES = [(0.11, -2.3), (0.11, 2.3)]  # (t, x ofseti)
EYE_R = 1.15


def render():
    n = SIZE * SS
    c = (np.arange(n) + 0.5) / SS  # alt piksel merkezleri
    yy, xx = np.meshgrid(c, c, indexing="ij")
    t = (yy - Y0) / LENGTH
    inside = (t >= 0) & (t <= 1)
    hw = np.where(inside, _half_width(np.clip(t, 0, 1)), 0.0)
    inside &= np.abs(xx - CX) < hw

    region = np.zeros((n, n), np.int64)
    region[inside & (t < HEAD_END)] = 1
    region[inside & (t >= HEAD_END) & (t < TAIL_START)] = 2
    region[inside & (t >= TAIL_START)] = 3
    for et, ex in EYES:
        ey = Y0 + et * LENGTH
        eye = (yy - ey) ** 2 + (xx - (CX + ex)) ** 2 < EYE_R**2
        region[inside & eye] = 4

    colors = np.zeros((5, 3))
    for i, k in enumerate(["head", "body", "tail", "eye"], start=1):
        colors[i] = PALETTE[k]
    rgb_sub = colors[region] * inside[..., None]

    def pool(a):
        return a.reshape(SIZE, SS, SIZE, SS, *a.shape[2:]).mean(axis=(1, 3))

    alpha = pool(inside.astype(np.float64))
    rgb_premult = pool(rgb_sub)

    # piksel başına baskın bölge ve eksen koordinatları (prob ve sayım için)
    onehot = np.stack([region == i for i in range(5)], -1).astype(np.float64)
    region_px = pool(onehot).argmax(-1)
    region_px[alpha < 0.5] = 0
    pc = np.arange(SIZE) + 0.5
    py, px = np.meshgrid(pc, pc, indexing="ij")
    axial = np.clip((py - Y0) / LENGTH, 0, 1)
    hw_px = np.maximum(_half_width(axial), 1e-3)
    lateral = np.clip((px - CX) / hw_px, -1, 1)

    rgba = np.concatenate([rgb_premult, alpha[..., None]], -1).astype(np.float32)
    return rgba, region_px, axial.astype(np.float32), lateral.astype(np.float32)


def main():
    out = Path(__file__).parent
    rgba, region, axial, lateral = render()
    a = rgba[..., 3:4]
    straight = np.where(a > 0, rgba[..., :3] / np.maximum(a, 1e-6), 0)
    img = np.concatenate([straight, a], -1)
    Image.fromarray((img * 255).round().astype(np.uint8), "RGBA").save(out / "planaria.png")
    big = Image.fromarray((img * 255).round().astype(np.uint8), "RGBA").resize(
        (SIZE * 8, SIZE * 8), Image.NEAREST
    )
    big.save(out / "planaria_x8.png")
    palette = np.array([PALETTE[k] for k in ["head", "body", "tail", "eye"]], np.float32)
    np.savez(
        out / "planaria.npz",
        rgba=rgba,
        region=region,
        axial=axial,
        lateral=lateral,
        palette=palette,
        palette_names=np.array(["head", "body", "tail", "eye"]),
    )
    counts = {k: int((region == i).sum()) for i, k in enumerate(["bg", "head", "body", "tail", "eye"])}
    print(f"hedef yazıldı: alfa>0.1 piksel={int((rgba[..., 3] > 0.1).sum())}, bölgeler={counts}")


if __name__ == "__main__":
    main()
