"""Eğitim: sample pool, hasar rejimleri (R0-R3), rastgele uzunlukta açılım ve BPTT.

Kullanım:
    python -m nca.train configs/r1.yaml --seed 0
    python -m nca.train configs/r3.yaml --seed 1 --set iterations=2000
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from torch.utils.checkpoint import checkpoint

from . import damage as dmg
from .metrics import batch_iou
from .model import CAModel, make_seed
from .utils import ROOT, device, load_config, load_target, pad_to, save_gif, save_grid, set_seed, to_uint8


def apply_damage(x, regime, gen, cfg):
    """x: hasar görecek örnekler (yerinde değiştirilir)."""
    if regime == "R0" or x.shape[0] == 0:
        return
    r_range = tuple(cfg.get("circle_r", (0.1, 0.4)))
    k_range = tuple(cfg.get("keep_frac", (0.2, 0.8)))
    for i in range(x.shape[0]):
        kind = regime
        if regime == "R3":
            kind = "R1" if torch.rand(1, generator=gen).item() < cfg.get("r3_circle_prob", 0.5) else "R2"
        if kind == "R1":
            dmg.circle_damage(x[i : i + 1], gen, r_range)
        elif kind == "R2":
            dmg.random_cut(x[i : i + 1], gen, k_range)
        else:
            raise ValueError(regime)


def run_steps(model, x, steps, segment):
    """`segment` adımlık parçalar halinde gradyan checkpoint'li açılım (bellek için)."""
    if segment <= 0:
        return model.rollout(x, steps)
    done = 0
    while done < steps:
        k = min(segment, steps - done)
        x = checkpoint(model.rollout, x, k, use_reentrant=False)
        done += k
    return x


def normalize_grads(model):
    """Parametre başına gradyan normalizasyonu: uzun BPTT açılımında patlamayı önler."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= p.grad.norm() + 1e-8


def train(cfg, out_dir=None):
    dev = device()
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 12345)
    torch.backends.cudnn.benchmark = True

    out_dir = Path(out_dir or ROOT / cfg.get("out_dir", "runs") / f"{cfg['name']}_s{cfg['seed']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshots").mkdir(exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    tgt = load_target(cfg.get("target"))
    size = cfg["grid"]
    target = pad_to(tgt["rgba"], size).to(dev)

    m = cfg["model"]
    model = CAModel(m["n_hidden"], m["hidden_dim"], m["perception"], m["fire_rate"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [cfg["lr_drop_iter"]], gamma=cfg["lr_drop"])

    seed = make_seed(1, size, model.channels, dev)
    p = cfg["pool"]
    use_pool = p["enabled"]
    pool = seed.repeat(p["size"], 1, 1, 1) if use_pool else None
    bs = cfg["batch_size"]
    dcfg = cfg["damage"]
    regime = dcfg["regime"]
    smin, smax = cfg["steps"]
    segment = cfg.get("checkpoint_segment", 0)

    log_f = open(out_dir / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "lr", "steps", "sec"])
    t0 = time.time()
    print(f"[{out_dir.name}] parametre={model.n_params()} cihaz={dev} rejim={regime}")

    for it in range(cfg["iterations"]):
        if use_pool:
            idx = torch.randint(p["size"], (bs,), generator=gen).to(dev)
            x = pool[idx].clone()
            with torch.no_grad():
                order = ((x[:, :4] - target[None]) ** 2).mean(dim=(1, 2, 3)).argsort(descending=True)
            x, idx = x[order], idx[order]
            x[:1] = seed  # en yüksek kayıplı örnek tohumla değişir
            if it >= dcfg.get("start_iter", 0):
                n = dcfg.get("n_damaged", 3)
                apply_damage(x[bs - n :], regime, gen, dcfg)
        else:
            x = seed.repeat(bs, 1, 1, 1)

        steps = int(torch.randint(smin, smax + 1, (1,), generator=gen))
        x = run_steps(model, x, steps, segment)
        loss = ((x[:, :4] - target[None]) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.get("grad_norm", "per_param") == "per_param":
            normalize_grads(model)
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

        if use_pool:
            pool[idx] = x.detach()

        if it % cfg.get("log_every", 50) == 0 or it == cfg["iterations"] - 1:
            log.writerow([it, f"{loss.item():.6f}", sched.get_last_lr()[0], steps, f"{time.time() - t0:.1f}"])
            log_f.flush()
            if it % (cfg.get("log_every", 50) * 10) == 0:
                print(f"[{out_dir.name}] it={it:5d} loss={loss.item():.5f} t={time.time() - t0:.0f}s")
        if it % cfg.get("snapshot_every", 1000) == 0 or it == cfg["iterations"] - 1:
            save_grid(out_dir / "snapshots" / f"batch_{it:05d}.png", to_uint8(x), scale=3)
            if use_pool:
                save_grid(out_dir / "snapshots" / f"pool_{it:05d}.png", to_uint8(pool[:16]), scale=2, ncol=8)

    log_f.close()
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config(), "config": cfg},
        out_dir / "model.pt",
    )
    summary = finish(model, target, size, out_dir, dev)
    summary["train_sec"] = round(time.time() - t0, 1)
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary))
    print(f"[{out_dir.name}] bitti: {summary}")
    return out_dir


@torch.no_grad()
def finish(model, target, size, out_dir, dev):
    """Eğitim sonu: büyüme GIF'i ve hızlı biçim metrikleri."""
    x = make_seed(16, size, model.channels, dev)
    frames, ious = [], {}
    for t in range(1, 1001):
        x = model(x)
        if t <= 300 and t % 2 == 0:
            frames.append(to_uint8(x[:1])[0])
        if t in (100, 200, 1000):
            ious[f"iou_{t}"] = round(batch_iou(x, target).mean().item(), 4)
    save_gif(out_dir / "growth.gif", frames, scale=4)
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="a.b=değer biçiminde config geçersiz kılma")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg, args.out)


if __name__ == "__main__":
    main()
