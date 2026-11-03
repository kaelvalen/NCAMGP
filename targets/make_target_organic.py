"""Planarya (Schmidtea mediterranea benzeri) hedef organizmayı prosedürel olarak çizer.

Gerçek planaryanın özellikleri: kulakçıklı üçgen baş, beyaz pigmentsiz halka içinde iki
göz lekesi, benekli kahverengi pigment, koyu sırt orta çizgisi, yarı saydam açık renkli kenar,
gövdenin ortasında soluk farinks ve üç dallı bağırsağın soluk gölgesi, sivrilen kuyruk.
Dış hat ve orta hat hafif kıvrımlıdır (sabit tohumlu düşük frekanslı dalgalanma); pigment
fBm gürültüsüyle beneklenir. Hepsi sabit tohumludur: hedef deterministiktir.

Baş sayımı renge değil göz lekelerine dayanır (nca/metrics.py): bu yüzden gözler hedefteki
tek çok koyu yapıdır; pigment hiçbir yerde göz kadar koyulaşmaz.

Çıktılar:
  targets/planaria_organic.png  düz (premultiplied olmayan) RGBA
  targets/planaria_organic_x8.png 8x büyütülmüş önizleme
  targets/planaria_organic.npz  rgba (premultiplied, float32), region (0 arka plan, 1 baş,
                             2 gövde, 3 kuyruk, 4 göz, 5 farinks), axial (baş ucu 0, kuyruk ucu 1),
                             lateral (orta hatta 0, kenarlarda +-1)
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, zoom

SIZE = 56
SS = 8  # alt örnekleme çarpanı
CX = SIZE / 2
Y0, LENGTH = 2.0, 52.0
SEED = 7

HEAD_END, TAIL_START = 0.2, 0.8
EYE_LUMA = 0.2  # gözler bu parlaklığın altında
PIGMENT_FLOOR = 0.27  # pigmentin en düşük parlaklığı (göz eşiğinin üstünde pay bırakır)

# (eksen konumu t, yarı genişlik piksel): sivri uç, kulakçıklar, boyun, gövde, kuyruk
_PROFILE = np.array(
    [
        (0.00, 1.2),
        (0.03, 4.0),
        (0.075, 7.6),
        (0.11, 6.9),
        (0.17, 5.7),
        (0.30, 7.6),
        (0.50, 8.3),
        (0.68, 7.6),
        (0.84, 5.0),
        (0.95, 2.2),
        (1.00, 0.5),
    ]
)
_half_width = PchipInterpolator(_PROFILE[:, 0], _PROFILE[:, 1])

EYES = [(0.085, -2.7), (0.085, 2.7)]
EYE_HALO, EYE_PUPIL = 1.8, 1.4


def fbm(shape, rng, octaves=4, base=6.0):
    """Değer gürültüsü tabanlı fBm, [0, 1]."""
    out = np.zeros(shape)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = max(2, int(base * 2**o))
        g = rng.random((n, n))
        up = zoom(g, (shape[0] / n, shape[1] / n), order=3)[: shape[0], : shape[1]]
        out += amp * up
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, w):
    return a * (1 - w[..., None]) + np.asarray(b) * w[..., None]


def render():
    rng = np.random.default_rng(SEED)
    n = SIZE * SS
    c = (np.arange(n) + 0.5) / SS
    yy, xx = np.meshgrid(c, c, indexing="ij")
    t = (yy - Y0) / LENGTH
    tc = np.clip(t, 0, 1)

    # hafif S kıvrımlı orta hat ve dalgalı kenar
    mid = CX + 0.9 * np.sin(np.pi * (1.1 * tc + 0.15)) - 0.9 * np.sin(np.pi * 0.15)
    wobble = 1 + 0.035 * np.sin(2 * np.pi * (3.2 * tc + 0.3)) + 0.02 * np.sin(2 * np.pi * (7.1 * tc + 1.1))
    hw = _half_width(tc) * wobble
    dx = xx - mid
    lat = dx / np.maximum(hw, 1e-3)  # orta hatta 0, kenarda +-1
    inside = (t >= 0) & (t <= 1) & (np.abs(lat) < 1)
    edge_dist = (1 - np.abs(lat)) * hw  # kenara uzaklık (piksel)

    # --- pigment ---
    speck = fbm((n, n), rng, octaves=4, base=SIZE / 5)
    fine = fbm((n, n), rng, octaves=2, base=SIZE / 1.6)
    body = np.array([0.40, 0.27, 0.16])
    rgb = np.broadcast_to(body, (n, n, 3)).copy()
    rgb *= (0.70 + 0.60 * speck)[..., None]  # benekler
    rgb *= (0.86 + 0.28 * fine)[..., None]
    rgb = mix(rgb, (0.22, 0.14, 0.085), 0.55 * np.exp(-((lat / 0.28) ** 2)))  # sırt çizgisi
    rgb = mix(rgb, (0.70, 0.56, 0.42), smoothstep(0.55, 1.0, np.abs(lat)) * 0.85)  # açık kenar

    # bağırsak: ön dal orta hatta, iki arka dal yanlarda (farinksin iki yanından kuyruğa)
    gut = np.exp(-((lat / 0.11) ** 2)) * smoothstep(0.16, 0.24, tc) * (1 - smoothstep(0.40, 0.46, tc))
    for side in (-1, 1):
        gut = np.maximum(gut, np.exp(-(((lat - side * 0.40) / 0.12) ** 2)) * smoothstep(0.42, 0.5, tc) * (1 - smoothstep(0.86, 0.93, tc)))
    rgb = mix(rgb, (0.60, 0.48, 0.36), 0.65 * gut)

    # farinks: gövde ortasında soluk uzun oval
    ph = ((tc - 0.56) / 0.105) ** 2 + (lat / 0.27) ** 2
    pharynx = 1 - smoothstep(0.75, 1.0, ph)
    rgb = mix(rgb, (0.80, 0.70, 0.58), 0.85 * pharynx)
    rgb = mix(rgb, (0.55, 0.44, 0.33), 0.5 * smoothstep(0.80, 0.95, ph) * (1 - smoothstep(0.95, 1.15, ph)))  # farinks kılıfı

    # baş: kulakçıklar biraz açık, kuyruk ucu biraz koyu
    rgb = mix(rgb, (0.62, 0.48, 0.36), 0.35 * smoothstep(0.6, 1.0, np.abs(lat)) * (1 - smoothstep(0.06, 0.16, tc)))
    rgb = mix(rgb, (0.27, 0.18, 0.11), 0.35 * smoothstep(0.85, 1.0, tc))

    # pigment tabanı: hiçbir doku piksel göz kadar koyu olmasın (göz sayımı için pay)
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    rgb *= np.maximum(1.0, PIGMENT_FLOOR / np.maximum(luma, 1e-3))[..., None]

    # gözler: pigmentsiz beyaz halka + koyu göz kabı (hafif içe ve öne kaymış)
    region_eye = np.zeros((n, n), bool)
    for et, ex in EYES:
        ey = Y0 + et * LENGTH
        ecx = np.interp(ey, c, mid[:, 0]) + ex
        d = np.hypot(yy - ey, xx - ecx)
        rgb = mix(rgb, (0.93, 0.90, 0.84), 1 - smoothstep(EYE_HALO - 0.4, EYE_HALO, d))
        dp = np.hypot(yy - (ey - 0.15), xx - (ecx - 0.25 * np.sign(ex)))
        pupil = 1 - smoothstep(EYE_PUPIL - 0.3, EYE_PUPIL, dp)
        rgb = mix(rgb, (0.03, 0.025, 0.03), pupil)
        region_eye |= pupil > 0.5

    # yarı saydam kenar: alfa kenarda yumuşakça düşer
    alpha_sub = inside * (0.55 + 0.45 * smoothstep(0.0, 1.6, edge_dist))
    rgb_sub = rgb * alpha_sub[..., None]

    region = np.zeros((n, n), np.int64)
    region[inside & (tc < HEAD_END)] = 1
    region[inside & (tc >= HEAD_END) & (tc < TAIL_START)] = 2
    region[inside & (tc >= TAIL_START)] = 3
    region[inside & (pharynx > 0.5)] = 5
    region[inside & region_eye] = 4

    def pool(a):
        return a.reshape(SIZE, SS, SIZE, SS, *a.shape[2:]).mean(axis=(1, 3))

    alpha = pool(alpha_sub)
    rgb_premult = pool(rgb_sub)
    onehot = np.stack([region == i for i in range(6)], -1).astype(np.float64)
    region_px = pool(onehot).argmax(-1)
    region_px[alpha < 0.3] = 0
    axial = np.clip(pool(t), 0, 1)
    lateral = np.clip(pool(np.where(inside, lat, 0.0)), -1, 1)

    rgba = np.concatenate([rgb_premult, alpha[..., None]], -1).astype(np.float32)
    return rgba, region_px, axial.astype(np.float32), lateral.astype(np.float32)


def main():
    out = Path(__file__).parent
    rgba, region, axial, lateral = render()
    a = rgba[..., 3:4]
    straight = np.where(a > 0, rgba[..., :3] / np.maximum(a, 1e-6), 0)
    img = (np.concatenate([straight, a], -1) * 255).round().astype(np.uint8)
    Image.fromarray(img, "RGBA").save(out / "planaria_organic.png")
    Image.fromarray(img, "RGBA").resize((SIZE * 8, SIZE * 8), Image.NEAREST).save(out / "planaria_organic_x8.png")
    np.savez(out / "planaria_organic.npz", rgba=rgba, region=region, axial=axial, lateral=lateral, eye_luma=np.float32(EYE_LUMA))
    names = ["bg", "head", "body", "tail", "eye", "pharynx"]
    counts = {k: int((region == i).sum()) for i, k in enumerate(names)}
    print(f"hedef yazıldı: {SIZE}x{SIZE}, alfa>0.1 piksel={int((rgba[..., 3] > 0.1).sum())}, bölgeler={counts}")


if __name__ == "__main__":
    main()
