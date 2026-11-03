"""Ata fiziğinin eğitimi: genom koşullu NCA, ata canlı uzayındaki (targets/creature_space.py)
her canlıyı tek hücreden büyütmeyi ve parçadan yenilemeyi öğrenir.

Hedefler önceden üretilmiş bir bankadan (n_bank genom) gelir; sample pool, dairesel hasar ve
yarım düzlem kesmesi, BPTT ve parametre başına gradyan normalizasyonu Mordvintsev (2020) ile
aynıdır. Kayıp: kendi genomunun hedefiyle RGBA MSE + değer sınırları.

Kullanım: python -m nca.train_ancestor configs/ancestor.yaml --seed 0
"""

import argparse
import csv
import sys
import time

import numpy as np
import torch
import yaml

from .evo import EvoCA, Physics, make_seed, step
from .train import normalize_grads
from .train_evo import apply_damage, run
from .utils import ROOT, device, load_config, save_grid, set_seed, to_uint8

sys.path.insert(0, str(ROOT / "targets"))
import creature_space as cs  # noqa: E402


def load_bank(n, size, seed=0):
    """Banka önbellekten okunur ya da üretilir; hedefler ızgaranın ortasına yerleştirilir."""
    path = ROOT / "runs" / f"ancestor_bank_{n}_{seed}.npz"
    if path.exists():
        d = np.load(path)
        g, t = d["g"], d["t"]
    else:
        print(f"banka üretiliyor: {n} canlı")
        g, t = cs.bank(n, seed)
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, g=g, t=t)
    pad = (size - t.shape[-1]) // 2
    t = np.pad(t, ((0, 0), (0, 0), (pad, size - t.shape[-1] - pad), (pad, size - t.shape[-1] - pad)))
    return torch.from_numpy(g), torch.from_numpy(t)


def bounds(x):
    rgba = x[:, :4]
    return (
        (torch.relu(rgba[:, :3] - rgba[:, 3:4]) ** 2).flatten(1).mean(1)
        + (torch.relu(-rgba) ** 2 + torch.relu(rgba - 1) ** 2).flatten(1).mean(1)
    ) * 10 + (torch.relu(x[:, 5:].abs() - 3) ** 2).flatten(1).mean(1)


def train(cfg):
    dev = device()
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 77)
    out = ROOT / cfg.get("out_dir", "runs") / f"{cfg['name']}_s{cfg['seed']}"
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    size, bs = cfg["grid"], cfg["batch_size"]
    G_bank, T_bank = load_bank(cfg["n_bank"], size, cfg.get("bank_seed", 0))
    G_bank, T_bank = G_bank.to(dev), T_bank.to(dev)
    m = cfg["model"]
    model = EvoCA(m["n_hidden"], m["n_genome"], m["hidden_dim"], m["perception"], m["fire_rate"], m.get("sense_v", True)).to(dev)
    phys = Physics(**cfg.get("physics", {}))
    if cfg.get("init_from"):  # kaldığı yerden devam (ör. daha uzun eğitim)
        model.load_state_dict(torch.load(ROOT / cfg["init_from"], map_location=dev, weights_only=False)["state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [cfg["lr_drop_iter"]], gamma=cfg["lr_drop"])

    P = cfg["pool_size"]
    pool_i = torch.randint(len(G_bank), (P,), generator=gen).to(dev)
    pool_x, pool_g = make_seed(G_bank[pool_i], size, model.channels)
    pool_loss = torch.full((P,), float("inf"), device=dev)
    smin, smax = cfg["steps"]
    dcfg = cfg["damage"]
    eval_i = torch.arange(8, device=dev)

    log_f = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "mse", "sec"])
    t0 = time.time()
    print(f"[{out.name}] parametre={model.n_params()} banka={len(G_bank)}", flush=True)
    for it in range(cfg["iterations"]):
        idx = torch.randint(P, (bs,), generator=gen).to(dev)
        idx = idx[pool_loss[idx].argsort(descending=True)]
        x, g, bi = pool_x[idx].clone(), pool_g[idx].clone(), pool_i[idx].clone()
        bi[:1] = torch.randint(len(G_bank), (1,), generator=gen).to(dev)  # en kötü örnek: yeni canlı
        x[:1], g[:1] = make_seed(G_bank[bi[:1]], size, model.channels)
        if it >= dcfg["start_iter"]:
            n = dcfg["n_damaged"]
            apply_damage(x[bs - n :], g[bs - n :], gen, dcfg, phys)
        steps = int(torch.randint(smin, smax + 1, (1,), generator=gen))
        x, g = run(model, x, g, phys, steps, cfg.get("checkpoint_segment", 16))
        mse = ((x[:, :4] - T_bank[bi]) ** 2).flatten(1).mean(1)
        total = mse + cfg.get("bounds_weight", 1.0) * bounds(x)
        loss = total.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        normalize_grads(model)
        opt.step()
        sched.step()
        pool_x[idx], pool_g[idx], pool_i[idx] = x.detach(), g.detach(), bi
        pool_loss[idx] = total.detach()
        if it % cfg.get("log_every", 50) == 0 or it == cfg["iterations"] - 1:
            log.writerow([it, f"{loss.item():.6f}", f"{mse.mean().item():.6f}", f"{time.time() - t0:.1f}"])
            log_f.flush()
        if it % cfg.get("print_every", 500) == 0:
            print(f"[{out.name}] it={it} mse={mse.mean().item():.5f} t={time.time() - t0:.0f}s", flush=True)
        if it % cfg.get("snapshot_every", 1000) == 0 or it == cfg["iterations"] - 1:
            with torch.no_grad():
                xe, ge = make_seed(G_bank[eval_i], size, model.channels)
                for _ in range(200):
                    xe, ge = step(model, xe, ge, phys)
            frames = [f for pair in zip(to_uint8(T_bank[eval_i]), to_uint8(xe[:, :4])) for f in pair]
            save_grid(out / "snapshots" / f"gallery_{it:05d}.png", frames, scale=2, ncol=8)
    log_f.close()
    torch.save({"state_dict": model.state_dict(), "model_config": model.config(), "physics": vars(phys), "config": cfg}, out / "model.pt")
    print(f"[{out.name}] bitti {time.time() - t0:.0f}s", flush=True)
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
