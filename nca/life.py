"""Yaşayan ekosistem fiziği (docs/superpowers/specs/2026-11-25-yasayan-ekosistem-design.md).

Durum x (B, C, H, W): [R, G, B, m, V, E, gizli_1..gizli_n]
  m: kütle (eski alfa; hücre m > 0,1 iken canlı), V: yara potansiyeli, E: enerji.
  Kural RGB ve gizli kanalları yazar; m, V, E'yi yalnız fizik değiştirir.
Genom g (B, G, H, W): FiLM ile kuralın gizli katmanını değiştirir (sabit rastgele izdüşümler).
Kaynak R (B, 1, H, W) ve yabancı kütle F (B, 1, H, W) kuralın algıladığı dünya alanlarıdır.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .evo import V_CH, update_potential
from .model import ALIVE_THRESHOLD, KERNELS, PERCEPTIONS

RGB = slice(0, 3)
M_CH = 3
E_CH = 5
H0 = 6  # ilk gizli kanal
H_MAX = 5.0  # gizli kanal sınırı (ön eğitimin sınır kaybı |h| <= 3 tutar)

def pad1(t, circular=False, value=0.0):
    """1 hücrelik kenar dolgusu: sarmal (toroidal arena) ya da sabit değer."""
    return F_.pad(t, (1, 1, 1, 1), mode="circular") if circular else F_.pad(t, (1, 1, 1, 1), value=value)


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

    def forward(self, x, R, F, g, circular=False):
        p = F_.conv2d(pad1(torch.cat([x, R, F], 1), circular), self.filters, groups=self.n_in)
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


# --- taşınım ---
def shift(t, dy, dx):
    """out[y, x] = t[y - dy, x - dx] (sıfır dolgulu): (dy, dx) yönünde bir hücre kaydırır."""
    H, W = t.shape[-2:]
    p = F_.pad(t, (1, 1, 1, 1))
    return p[..., 1 - dy : 1 - dy + H, 1 - dx : 1 - dx + W]


def neighbor(t, dy, dx):
    """out[y, x] = t[y + dy, x + dx]: kaynağın (dy, dx) yönündeki hedefin değeri."""
    return shift(t, -dy, -dx)


_NB = [(dy + 1) * 3 + (dx + 1) for dy, dx in DIRS]  # unfold'da (dy, dx) komşusunun indeksi
_OPP = [DIRS.index((-dy, -dx)) for dy, dx in DIRS]  # ters yön
_GATHER = {}


def neighbors8(t, circular=False):
    """(B, C, H, W) -> (B, C, 8, H, W): out[:, :, k, y, x] = t[y + dy_k, x + dx_k] (sıfır dolgulu)."""
    B, C, H, W = t.shape
    return F_.unfold(pad1(t, circular), 3).view(B, C, 9, H, W)[:, :, _NB]


def gather_flow(P, circular=False):
    """P (B, C, 8, H, W): kaynak hücreden k yönüne giden pay -> hedefte toplanan (B, C, H, W).
    Tek gruplanmış konvolüsyon: hedef (y, x), k payını (y - dy_k, x - dx_k) kaynağından alır."""
    B, C, K, H, W = P.shape
    key = (C, P.device, P.dtype)
    if key not in _GATHER:
        w = torch.zeros(C, K, 3, 3, dtype=P.dtype, device=P.device)
        for k, (dy, dx) in enumerate(DIRS):
            w[:, k, 1 - dy, 1 - dx] = 1
        _GATHER[key] = w
    return F_.conv2d(pad1(P.reshape(B, C * K, H, W), circular), _GATHER[key], groups=C)


def transport(m, flow, ext, inten, tag=None, circular=False):
    """Kütle korunumlu taşınım. Her hücre kütlesini `flow` kesirlerine göre (0 = kal, 1..8 = DIRS)
    komşularına dağıtır; `ext` (yaygın) kütle gibi taşınır, `inten` (yoğun) kütle ağırlıklı ortalanır.
    Izgara dışına giden pay yerinde kalır (circular: kenar sarmaldır, pay karşı kenara geçer). `tag` verilirse bireyler kaynaşmaz: başka bireyin canlı
    hücresine ya da canlı komşuları arasında iki farklı birey bulunan boş hücreye giden pay geri döner.
    Döndürür (m2, ext2, inten2, tag2); tag2, tag None ise None."""
    with torch.no_grad():
        if tag is None:
            allowed = neighbors8(torch.ones_like(m), circular)[:, 0]  # ızgara içi
        else:
            alive = m > 0  # eşik altı ince kütle de dolu sayılır: iki birey tek hücrede karışmaz
            big = torch.tensor(1e9, dtype=m.dtype, device=m.device)
            tmax = F_.max_pool2d(pad1(torch.where(alive, tag, -big), circular, -1e9), 3, 1)  # canlı komşu etiketlerinin
            tmin = -F_.max_pool2d(pad1(torch.where(alive, -tag, -big), circular, -1e9), 3, 1)  # en büyüğü / en küçüğü
            any_alive = (tmax > -big / 2).to(m.dtype)
            nb = neighbors8(torch.cat([torch.ones_like(m), tag, alive.to(m.dtype), tmax, tmin, any_alive], 1), circular)
            inside, t_tag, occupied, n_max, n_min, n_any = nb.unbind(1)
            same = (t_tag - tag).abs() < 1e-4
            # boş hedef: canlı komşuları yalnız bu bireyden olmalı (iki bireyin arasına doğum yok)
            clean = (n_any == 0) | (((n_max - tag).abs() < 1e-4) & ((n_min - tag).abs() < 1e-4))
            allowed = ((inside > 0) & torch.where(occupied > 0, same, clean)).to(m.dtype)
    move = flow[:, 1:] * allowed  # (B, 8, H, W)
    stay = flow[:, :1] + (flow[:, 1:] * (1 - allowed)).sum(1, keepdim=True)  # 1 - Σmove yuvarlamada negatife düşebilir
    q = torch.cat([m, ext, inten * m], 1)  # kütle, yaygın, kütle x yoğun
    q2 = q * stay + gather_flow(q[:, :, None] * move[:, None], circular)
    ne = ext.shape[1]
    m2, ext2, num = q2[:, :1], q2[:, 1 : 1 + ne], q2[:, 1 + ne :]
    inten2 = torch.where(m2 > 1e-8, num / m2.clamp(min=1e-8), torch.zeros_like(num))
    tag2 = None
    if tag is not None:
        with torch.no_grad():  # katkı verenlerin etiketi (engel sayesinde hepsi aynı)
            sent = tag[:, :, None] * ((m[:, :, None] * move[:, None]) > 0)  # (B, 1, 8, H, W)
            inc = neighbors8(sent[:, 0], circular)  # (B, 8 yön, 8 komşu, H, W)
            k = torch.arange(8, device=m.device)
            incoming = inc[:, k, torch.tensor(_OPP, device=m.device)]  # k yönüne gidenin hedefe gelişi
            tag2 = torch.maximum(tag * ((m * stay) > 0), incoming.amax(1, keepdim=True))
    return m2, ext2, inten2, tag2


# --- adım fiziği ---
@dataclass
class LifePhysics:
    """Elle yazılmış fizik sabitleri. Besin tek bir para birimidir: R (kaynak), m (kütle), E (enerji)."""

    v_decay: float = 0.97
    v_gain: float = 1.0
    v_diffuse: float = 0.2
    v_max: float = 2.0
    kappa: float = 0.02  # aynı birey içinde genom homojenleşmesi (dünyada uygulanır)
    k_up: float = 0.1  # hücre adım başına yerel R'nin bu oranını alır (kütleyle, en çok 1'e kadar)
    c_m: float = 0.002  # kütle başına bakım
    c_move: float = 0.01  # yerinden giden kütle başına hareket maliyeti
    r_g: float = 0.05  # büyüme: enerjiyi kütleye çevirme hızı
    r_diffuse: float = 0.25  # kaynak difüzyonu (korunumlu, 4 komşu)
    k_eat: float = 0.2
    eta: float = 0.6
    c_attack: float = 0.005
    torus: bool = False  # sarmal arena (ön eğitim: duvar yok); dünya kenarlıdır


def diffuse(R, d, circular=False):
    if circular:  # sarmal arena: her hücrenin 4 komşusu var
        nb = sum(torch.roll(R, s, dim) for s in (1, -1) for dim in (2, 3))
        return R + d / 4 * (nb - 4 * R)
    """Korunumlu difüzyon: komşu çiftleri arasında simetrik alışveriş, ızgara kenarından akı yok."""
    inside = torch.ones_like(R)
    acc = torch.zeros_like(R)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        acc = acc + neighbor(inside, dy, dx) * (neighbor(R, dy, dx) - R)
    return R + d / 4 * acc


def life_mask(m, circular=False):
    """Kendisi veya 3x3 komşularından biri canlı (Mordvintsev canlılık kuralı)."""
    return F_.max_pool2d(pad1(m, circular), 3, stride=1) > ALIVE_THRESHOLD


def life_step(model, x, g, R, phys, tag=None, F=None, fire_rate=None):
    """Bir adım: kural, büyüme ve hareket maliyeti (kaynak hücrede), taşınım, kaynak alımı, bakım,
    ölüm (besin R'ye döner), yara potansiyeli, kaynak difüzyonu. ΣR + Σm + ΣE korunur.
    Döndürür (x, g, R, tag, out); out: kural çıktıları (ateşleme maskeli) + "moved"."""
    fire_rate = model.fire_rate if fire_rate is None else fire_rate
    m, E = x[:, M_CH : M_CH + 1], x[:, E_CH : E_CH + 1]
    was_alive = m > ALIVE_THRESHOLD
    F = torch.zeros_like(m) if F is None else F
    out = model(x, R, F, g, phys.torus)
    fm = ((torch.rand_like(m) <= fire_rate) & (m > 0)).to(x.dtype)
    stay1 = torch.zeros_like(out["flow"])
    stay1[:, 0] = 1
    flow = out["flow"] * fm + stay1 * (1 - fm)
    out = {**out, "flow": flow, "grow": out["grow"] * fm, "attack": out["attack"] * fm}
    # kuralın yazdığı kanallar sınırlıdır: aşırı (mutant) genomlarda geri beslemeyle taşıp NaN üretmesin
    rgb = (x[:, RGB] + out["drgb"] * fm).clamp(0, 1)
    h = (x[:, H0:] + out["dh"] * fm).clamp(-H_MAX, H_MAX)
    # büyüme ve hareket maliyeti kaynak hücrede, eldeki enerjiyle sınırlı (E negatife düşmez)
    Ep = E.clamp(min=0)
    dm = phys.r_g * out["grow"] * Ep
    moved = m * (1 - flow[:, :1])
    cmove = torch.minimum(phys.c_move * moved, Ep - dm)
    m, E, R = m + dm, E - dm - cmove, R + cmove
    nh = h.shape[1]
    # g = None: genomsuz (ön eğitim, g = 0 ve FiLM kapalı); 32 sıfır kanalı taşımak belleği boşa harcar
    inten = torch.cat([rgb, h] if g is None else [rgb, h, g], 1)
    m, E, inten, tag = transport(m, flow, E, inten, tag, phys.torus)
    rgb, h = inten[:, :3], inten[:, 3 : 3 + nh]
    g = None if g is None else inten[:, 3 + nh :]
    u = torch.minimum(R, phys.k_up * R * m.clamp(0, 1))
    c = phys.c_m * m
    R, E = R - u + c, E + u - c
    # ölüm: açlık (E ≤ 0) ya da canlı komşusu olmayan ince kütle; besin R'ye döner. E < 0 artığı
    # aynı adımda R'ye eklenen bakımın karşılanmayan kısmıdır, R negatife düşmez.
    dead = (m > 0) & ((E <= 0) | ~life_mask(m, phys.torus))
    df = dead.to(x.dtype)
    R = R + (m + E) * df
    keep = 1 - df
    m, E, rgb, h = m * keep, E * keep, rgb * keep, h * keep
    with torch.no_grad():
        if g is not None:
            g = g * keep
        if tag is not None:
            tag = tag * keep
    xn = torch.cat([rgb, m, x[:, V_CH : V_CH + 1], E, h], 1)
    with torch.no_grad():  # shortcut: V'nin yayılımı sarmal arenada kenarda sarmaz (yalnız algı), kenarda iz görülürse sarmallaştır
        V = update_potential(xn, was_alive, m > ALIVE_THRESHOLD, phys)
    xn = torch.cat([rgb, m, V, E, h], 1)
    R = diffuse(R, phys.r_diffuse, phys.torus)
    out["moved"], out["uptake"] = moved, u
    return xn, g, R, tag, out


# --- yeme ---
def _foreign_links(x, tag):
    """Her yön için (B, 1, H, W) maske: hücre canlı, o yöndeki komşu canlı ve başka bireyden."""
    m = x[:, M_CH : M_CH + 1]
    alive = (m > ALIVE_THRESHOLD).to(x.dtype)
    links = []
    for dy, dx in DIRS:
        other = (neighbor(tag, dy, dx) - tag).abs() > 1e-4
        links.append(alive * neighbor(alive, dy, dx) * other.to(x.dtype))
    return links


def foreign_mass(x, tag):
    """3x3 komşulukta başka bireylere ait canlı hücrelerin kütle toplamı (kuralın F girdisi)."""
    m = x[:, M_CH : M_CH + 1]
    return sum(link * neighbor(m, dy, dx) for link, (dy, dx) in zip(_foreign_links(x, tag), DIRS))


@torch.no_grad()
def eat(x, R, tag, attack, phys):
    """Saldırısı a olan hücre başka bireyin komşu hücrelerinden besin alır. Kurban v, saldırganlarının
    toplam saldırısı A ile m_v·min(1, k_eat·A) kütle ve aynı oranda enerji kaybeder; ganimet
    saldırganlara a_j/A oranında dağılır: η payı E'ye, kalanı R'ye. Saldırı maliyeti c_attack·a
    (eldeki enerjiyle sınırlı) R'ye gider. Kurbanın yarasında potansiyel oluşur.
    Döndürür (x, R, lost, gain); lost: hücre başına yenilen kütle, gain: saldırganın aldığı ganimet
    (η'dan önce; E'ye η·gain girer)."""
    m, E = x[:, M_CH : M_CH + 1], x[:, E_CH : E_CH + 1]
    links = _foreign_links(x, tag)
    A = sum(link * neighbor(attack, dy, dx) for link, (dy, dx) in zip(links, DIRS))
    frac = (phys.k_eat * A).clamp(max=1)
    lost, eloss = m * frac, E.clamp(min=0) * frac
    per_a = torch.where(A > 0, (lost + eloss) / A.clamp(min=1e-8), torch.zeros_like(A))
    gain = attack * sum(link * neighbor(per_a, dy, dx) for link, (dy, dx) in zip(links, DIRS))
    E = E - eloss + phys.eta * gain
    R = R + (1 - phys.eta) * gain
    cost = torch.minimum(phys.c_attack * attack, E.clamp(min=0)) * (m > ALIVE_THRESHOLD)
    E, R = E - cost, R + cost
    m = m - lost
    V = (x[:, V_CH : V_CH + 1] + phys.v_gain * frac).clamp(max=phys.v_max)
    x = torch.cat([x[:, :M_CH], m, V, E, x[:, E_CH + 1 :]], 1)
    return x, R, lost, gain
