"""Biçim, onarım, birey ve baş sayımı metrikleri."""

import numpy as np
import torch
from scipy import ndimage
from skimage.measure import label, regionprops

from .model import ALIVE_THRESHOLD


def alive_np(x):
    """(C, H, W) tensör veya (H, W, C) dizi -> canlı maske (H, W) numpy."""
    if isinstance(x, torch.Tensor):
        return (x[3] > ALIVE_THRESHOLD).cpu().numpy()
    return x[..., 3] > ALIVE_THRESHOLD


def iou(pred, target):
    """İki bool maskenin IoU'su; torch (B,H,W) ise örnek başına vektör döndürür."""
    if isinstance(pred, torch.Tensor):
        inter = (pred & target).flatten(1).sum(1).float()
        union = (pred | target).flatten(1).sum(1).float()
        return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return 1.0 if union == 0 else inter / union


def batch_iou(x, target_rgba):
    """x: (B, C, H, W), target_rgba: (4, H, W). Alfa>0.1 maskelerinin IoU'su."""
    return iou(x[:, 3] > ALIVE_THRESHOLD, (target_rgba[3] > ALIVE_THRESHOLD)[None])


def rgba_mse(x, target_rgba):
    return ((x[:, :4] - target_rgba[None]) ** 2).mean(dim=(1, 2, 3))


def center_of_mass(mask):
    return np.array(ndimage.center_of_mass(mask)) if mask.any() else None


def aligned_iou(mask, target_mask):
    """Ağırlık merkezleri çakışacak şekilde tam sayı kaydırma sonrası IoU (yer değiştirmeye duyarsız)."""
    cm, ct = center_of_mass(mask), center_of_mass(target_mask)
    if cm is None:
        return 0.0
    dy, dx = np.round(ct - cm).astype(int)
    shifted = ndimage.shift(mask.astype(np.uint8), (dy, dx), order=0, mode="constant") > 0
    return iou(shifted, target_mask)


def place(target_mask, shape):
    """Hedef maskeyi verilen ızgaranın ortasına yerleştirir."""
    out = np.zeros(shape, bool)
    th, tw = target_mask.shape
    y0, x0 = (shape[0] - th) // 2, (shape[1] - tw) // 2
    out[y0 : y0 + th, x0 : x0 + tw] = target_mask
    return out


def components(mask, min_area=20):
    """Alfa maskesindeki bağlı bileşenler (8-komşuluk), alanı min_area altındakiler atılır."""
    lab = label(mask, connectivity=2)
    return [r for r in regionprops(lab) if r.area >= min_area], lab


def count_individuals(mask, min_area=20):
    return len(components(mask, min_area)[0])


def component_ious(mask, target_mask, min_area=20):
    """Her bileşen için hizalanmış IoU (bileşen kendi başına bir birey sayılır)."""
    comps, lab = components(mask, min_area)
    shape = (max(mask.shape[0], target_mask.shape[0]), max(mask.shape[1], target_mask.shape[1]))
    tgt = place(target_mask, shape)
    out = []
    for r in comps:
        m = np.zeros(shape, bool)
        m[: lab.shape[0], : lab.shape[1]] = lab == r.label
        out.append(aligned_iou(m, tgt))
    return out


def classify_pixels(state_np, palette):
    """(H, W, C) durum -> piksel sınıfı: -1 ölü, 0 baş, 1 gövde, 2 kuyruk, 3 göz (en yakın renk)."""
    a = state_np[..., 3]
    rgb = state_np[..., :3] / np.maximum(a[..., None], 1e-3)  # premultiplied -> düz renk
    d = ((rgb[..., None, :] - palette[None, None]) ** 2).sum(-1)
    cls = d.argmin(-1)
    cls[a <= 0.5] = -1  # yalnız belirgin dokuyu say
    return cls


def count_heads(state_np, palette, min_area=12):
    """Baş rengine (baş veya göz sınıfı) yakın piksellerin bağlı bileşen sayısı."""
    cls = classify_pixels(state_np, palette)
    head = (cls == 0) | (cls == 3)
    lab = label(head, connectivity=2)
    return sum(1 for r in regionprops(lab) if r.area >= min_area)


def count_tails(state_np, palette, min_area=10):
    cls = classify_pixels(state_np, palette)
    lab = label(cls == 2, connectivity=2)
    return sum(1 for r in regionprops(lab) if r.area >= min_area)


OUTCOMES = ["tek_bas", "iki_bas", "ikiz", "bassiz", "bozulma"]


def classify_outcome(state_np, palette, target_area, head_min_area=12):
    """Kutupluluk deneyi sonuç sınıfı.

    tek_bas  bir baş
    iki_bas  iki baş, kuyruk yok (Janus; Durant vd. 2017'nin iki başlı solucanı)
    ikiz     iki baş ve en az bir kuyruk: yaradan ikinci tam birey çıkmış, gövdeler bağlı
    bassiz   baş yok
    bozulma  alan aşırı küçük/büyük, birden fazla bileşen veya >2 baş
    """
    alive = state_np[..., 3] > ALIVE_THRESHOLD
    area = alive.sum()
    if area < 0.4 * target_area or area > 3.0 * target_area:
        return "bozulma"
    if count_individuals(alive) != 1:
        return "bozulma"
    h = count_heads(state_np, palette, head_min_area)
    if h == 2:
        return "iki_bas" if count_tails(state_np, palette) == 0 else "ikiz"
    return {0: "bassiz", 1: "tek_bas"}.get(h, "bozulma")


def first_hit(series, threshold=0.9):
    """series (T, B): her örnek için eşiği ilk geçtiği adım (geçmezse -1)."""
    hit = series >= threshold
    any_hit = hit.any(0)
    idx = hit.float().argmax(0)
    return torch.where(any_hit, idx, torch.full_like(idx, -1))


def settle_time(series, threshold=0.9):
    """series (T, B): eşiğin üstünde kalıcı olarak kalmaya başladığı adım (1 tabanlı);
    son adımda eşiğin altındaysa -1. Onarım başarısı = settle_time >= 0, yani pencere
    sonunda IoU >= eşik; geçici dokunuşlar başarı sayılmaz."""
    below = series < threshold
    t = series.shape[0]
    rev = below.flip(0).float()
    last_below = t - 1 - rev.argmax(0)  # son altında kalınan adım indeksi
    out = torch.where(below.any(0), last_below + 1, torch.zeros_like(last_below))
    return torch.where(below[-1], torch.full_like(out, -1), out + 1)
