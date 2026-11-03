"""Evrimsel fiziğin hedefsiz eğitimi (docs/evrim_tasarim.md, "Fiziğin hedefsiz eğitimi").

Hedef görüntü yoktur. Her örnek rastgele bir genomla tek hücreden büyür; kayıp:
  izometri fenotip mesafesi (kaba siluet + renk) genom mesafesiyle orantılı: görünür çeşitlilik
           ve küçük mutasyon -> küçük değişiklik
  kimlik   çözücü ağlar büyümüş canlının görüntüsünden (RGBA) ve şeklinden (alfa) genomu
           tahmin eder; NCA ve çözücüler birlikte bu hatayı küçültür -> farklı genom farklı canlı
  alan     canlı kütle [area_min, area_max] içinde
  tek      en büyük bileşen dışındaki kütle
  durağan  son `stable_k` adımdaki RGBA değişimi (homeostaz)
  yapı     beden içi renk standart sapmasının alt sınırı (düz leke olmasın)
  sınır    premultiplied tutarlılık ve değer sınırları
Sample pool + dairesel hasar ve yarım düzlem kesmesi: parça kendi genomunun canlısına dönmeli.

Kullanım: python -m nca.train_evo configs/evo.yaml --seed 0
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import ndimage

from . import damage as dmg
from .evo import EvoCA, Physics, alive_cells, make_seed, step, wound
from .train import normalize_grads
from .utils import ROOT, device, load_config, save_grid, set_seed, to_uint8


class Decoder(nn.Module):
    """Görüntüden genom tahmin eden küçük CNN (çeviriye dayanıklı: global havuzlama)."""

    def __init__(self, cin, n_genome, width=32):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, 2, 1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(4 * w, 64), nn.ReLU(), nn.Linear(64, n_genome))

    def forward(self, x):
        h = self.net(x)
        return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))


def augment(img, shift=3, noise=0.03):
    """Rastgele kaydırma ve gürültü: çözücü ince piksel ayrıntısı yerine biçime bakmalı."""
    dy, dx = np.random.randint(-shift, shift + 1, 2)
    img = torch.roll(img, (int(dy), int(dx)), (2, 3))
    return img + noise * torch.randn_like(img)


def outside_largest(alpha):
    """(B, H, W) numpy alfa -> en büyük bileşen dışındaki canlı hücre maskesi."""
    out = np.zeros(alpha.shape, bool)
    for i, a in enumerate(alpha):
        lab, n = ndimage.label(a > 0.1, structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            out[i] = (lab > 0) & (lab != 1 + int(np.argmax(sizes)))
    return out


def phenotype(rgba, cell=5):
    """Görünür fenotip gömmesi: kaba siluet (cell x cell havuzlanmış alfa) + gövde ortalama rengi."""
    a = rgba[:, 3:4].clamp(0, 1)
    sil = F.avg_pool2d(a, cell).flatten(1)
    w = a.flatten(2).sum(2).clamp(min=1)
    colour = (rgba[:, :3].clamp(0, 1).flatten(2).sum(2) / w)  # premultiplied ortalama ~ düz renk ortalaması
    return sil, colour


def isometry(rgba, genomes, cfg):
    """Fenotip mesafesi genom mesafesiyle orantılı olsun: rastgele genomlar görünür biçimde farklı,
    küçük mutasyonlar küçük değişiklik (düzgün uygunluk yüzeyi). Batch içindeki tüm çiftler."""
    sil, colour = phenotype(rgba)
    iw = cfg["isometry"]
    dp = torch.cdist(sil, sil) / math.sqrt(sil.shape[1]) + iw["colour_weight"] * torch.cdist(colour, colour) / math.sqrt(3)
    dg = torch.cdist(genomes, genomes) / math.sqrt(2 * genomes.shape[1])  # rastgele çiftte ~1
    iu = torch.triu_indices(len(genomes), len(genomes), 1, device=rgba.device)
    err = (dp - iw["scale"] * dg)[iu[0], iu[1]] ** 2
    per = torch.zeros(len(genomes), device=rgba.device)
    per.index_add_(0, iu[0], err)
    per.index_add_(0, iu[1], err)
    return per / (len(genomes) - 1) * iw["weight_scale"]


def losses(x, x_prev, genomes, dec, dec_s, cfg):
    """Örnek başına kayıp terimleri (dict of (B,))."""
    rgba = x[:, :4]
    a = rgba[:, 3].clamp(0, 1)
    # yumuşak hücre sayımı: soluk ama geniş doku alanı ucuza doldurmasın
    area = torch.sigmoid((rgba[:, 3] - 0.1) * 40).flatten(1).sum(1)
    w = cfg["loss"]
    amin, amax = cfg["area"]
    L = {}
    img = rgba.clamp(0, 1)
    L["kimlik"] = ((dec(augment(img)) - genomes) ** 2).mean(1)
    L["sekil"] = ((dec_s(augment(img[:, 3:4])) - genomes) ** 2).mean(1)
    L["izometri"] = isometry(rgba, genomes, cfg)
    L["alan"] = ((F.relu(amin - area) + F.relu(area - amax)) / amax) ** 2
    other = torch.from_numpy(outside_largest(a.detach().cpu().numpy())).to(x.device)
    L["tek"] = (a * other).flatten(1).sum(1) / area.clamp(min=1)
    L["duragan"] = ((rgba - x_prev[:, :4]) ** 2).flatten(1).mean(1) * 100
    body = (a > 0.5).float()
    n = body.flatten(1).sum(1).clamp(min=1)
    rgb = rgba[:, :3] / a[:, None].clamp(min=0.5)
    mu = (rgb * body[:, None]).flatten(2).sum(2) / n[:, None]
    var = (((rgb - mu[..., None, None]) ** 2) * body[:, None]).flatten(2).sum(2) / n[:, None]
    L["yapi"] = F.relu(w["min_color_std"] - (var.mean(1) + 1e-6).sqrt()) ** 2 * 10
    alive = (a > 0.1).float().detach()
    na = alive.flatten(1).sum(1).clamp(min=1)
    L["opak"] = ((1 - a) ** 2 * alive).flatten(1).sum(1) / na  # canlı hücre dolu olsun
    lum = (rgb * torch.tensor([0.299, 0.587, 0.114], device=x.device)[None, :, None, None]).sum(1)
    L["renk"] = (F.relu(lum - w["max_luma"]) ** 2 * body).flatten(1).sum(1) / n * 10  # beyaz hayalet olmasın
    edge = torch.ones_like(a)
    eb = w.get("edge_band", 3)
    edge[:, eb:-eb, eb:-eb] = 0
    L["kenar"] = (a * edge).flatten(1).sum(1) / area.clamp(min=1)  # ızgara kenarına yapışmasın
    # bilateral simetri: tohumdan geçen dikey eksene göre ayna (i -> W - i)
    mirror = torch.roll(torch.flip(rgba, [-1]), 1, -1)
    L["simetri"] = ((rgba - mirror) ** 2).flatten(1).sum(1) / area.clamp(min=1)
    L["sinir"] = (
        (F.relu(rgba[:, :3] - rgba[:, 3:4]) ** 2).flatten(1).mean(1)
        + (F.relu(-rgba) ** 2 + F.relu(rgba - 1) ** 2).flatten(1).mean(1)
    ) * 10 + (F.relu(x[:, 5:].abs() - 3) ** 2).flatten(1).mean(1)
    total = sum(w.get(k, 1.0) * v for k, v in L.items())
    return total, L


def run(model, x, g, phys, steps, segment):
    """Gradyan checkpoint'li açılım (bellek). x, g'yi döndürür."""
    from torch.utils.checkpoint import checkpoint

    def seg(x, g, k):
        for _ in range(k):
            x, g = step(model, x, g, phys)
        return x, g

    done = 0
    while done < steps:
        k = min(segment, steps - done) if segment > 0 else steps - done
        x, g = checkpoint(seg, x, g, k, use_reentrant=False) if segment > 0 else seg(x, g, k)
        done += k
    return x, g


def apply_damage(x, g, gen, cfg, phys):
    for i in range(x.shape[0]):
        before = alive_cells(x[i : i + 1])
        if torch.rand(1, generator=gen).item() < 0.5:
            dmg.circle_damage(x[i : i + 1], gen, tuple(cfg["circle_r"]), on_alive=True)
        else:
            dmg.halfplane_cut(x[i], float(torch.rand(1, generator=gen)) * 2 * math.pi,
                              float(torch.rand(1, generator=gen)) * (cfg["keep_frac"][1] - cfg["keep_frac"][0]) + cfg["keep_frac"][0])
        removed = before & ~alive_cells(x[i : i + 1])
        g[i] *= alive_cells(x[i : i + 1])[0].float()
        wound(x[i : i + 1], removed, phys)


@torch.no_grad()
def gallery(model, phys, genomes, size, steps=200):
    x, g = make_seed(genomes, size, model.channels)
    for _ in range(steps):
        x, g = step(model, x, g, phys)
    return x


def train(cfg):
    dev = device()
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 4242)
    out = ROOT / cfg.get("out_dir", "runs") / f"{cfg['name']}_s{cfg['seed']}"
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    m = cfg["model"]
    model = EvoCA(m["n_hidden"], m["n_genome"], m["hidden_dim"], m["perception"], m["fire_rate"], m.get("sense_v", True)).to(dev)
    phys = Physics(**cfg.get("physics", {}))
    G, size, bs = m["n_genome"], cfg["grid"], cfg["batch_size"]
    dec, dec_s = Decoder(4, G).to(dev), Decoder(1, G).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [cfg["lr_drop_iter"]], gamma=cfg["lr_drop"])
    opt_d = torch.optim.Adam(list(dec.parameters()) + list(dec_s.parameters()), lr=cfg["lr_decoder"])

    P = cfg["pool_size"]
    pool_gv = torch.randn(P, G, device=dev)
    pool_x, pool_g = make_seed(pool_gv, size, model.channels)
    pool_loss = torch.full((P,), float("inf"), device=dev)
    eval_genomes = torch.randn(16, G, generator=torch.Generator().manual_seed(0)).to(dev)
    smin, smax = cfg["steps"]
    k_st = cfg["stable_k"]
    dcfg = cfg["damage"]

    log_f = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    keys = ["kimlik", "sekil", "izometri", "alan", "tek", "duragan", "yapi", "opak", "renk", "kenar", "simetri", "sinir"]
    log.writerow(["iter", "loss", *keys, "sec"])
    t0 = time.time()
    print(f"[{out.name}] parametre={model.n_params()} çözücü={sum(p.numel() for p in dec.parameters())}")
    for it in range(cfg["iterations"]):
        idx = torch.randint(P, (bs,), generator=gen).to(dev)
        order = pool_loss[idx].argsort(descending=True)
        idx = idx[order]
        x, g, gv = pool_x[idx].clone(), pool_g[idx].clone(), pool_gv[idx].clone()
        # en kötü örnek yeni rastgele genomlu tohumla değişir
        gv[:1] = torch.randn(1, G, generator=gen).to(dev)
        x[:1], g[:1] = make_seed(gv[:1], size, model.channels)
        if it >= dcfg["start_iter"]:
            n = dcfg["n_damaged"]
            apply_damage(x[bs - n :], g[bs - n :], gen, dcfg, phys)

        steps = int(torch.randint(smin, smax + 1, (1,), generator=gen))
        x, g = run(model, x, g, phys, steps - k_st, cfg.get("checkpoint_segment", 16))
        x_prev = x.detach()
        x, g = run(model, x, g, phys, k_st, 0)
        total, L = losses(x, x_prev, gv, dec, dec_s, cfg)
        loss = total.mean()

        opt.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        loss.backward()
        normalize_grads(model)
        opt.step()
        opt_d.step()
        sched.step()

        pool_x[idx], pool_g[idx], pool_gv[idx] = x.detach(), g.detach(), gv
        pool_loss[idx] = total.detach()

        if it % cfg.get("log_every", 50) == 0 or it == cfg["iterations"] - 1:
            log.writerow([it, f"{loss.item():.5f}", *[f"{L[k].mean().item():.5f}" for k in keys], f"{time.time() - t0:.1f}"])
            log_f.flush()
        if it % cfg.get("print_every", 500) == 0:
            print(f"[{out.name}] it={it} " + " ".join(f"{k}={L[k].mean().item():.3f}" for k in keys) + f" t={time.time() - t0:.0f}s", flush=True)
        if it % cfg.get("snapshot_every", 500) == 0 or it == cfg["iterations"] - 1:
            save_grid(out / "snapshots" / f"gallery_{it:05d}.png", to_uint8(gallery(model, phys, eval_genomes, size)), scale=2, ncol=8)
            save_grid(out / "snapshots" / f"batch_{it:05d}.png", to_uint8(x[:, :4]), scale=2)

    log_f.close()
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config(), "physics": vars(phys), "config": cfg,
         "decoder": dec.state_dict(), "decoder_shape": dec_s.state_dict()},
        out / "model.pt",
    )
    print(f"[{out.name}] bitti {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
