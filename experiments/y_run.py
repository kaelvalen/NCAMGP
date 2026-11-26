"""Yaşayan ekosistem koşusu (Y2-Y4 ham verisi). Ön eğitilmiş ata, g = 0 kurucularla 192² dünyada;
dış kesmeler anneal_steps boyunca söner.

Koşullar: mu1 (config'teki μ), mu0 (μ = 0, kalıtsal varyasyonsuz kontrol).
Çıktı: runs/lifeworld_<koşul>_s<tohum>/ history.csv, events.csv, orgs.json, summary.yaml, snap_<t>.png
Tükenmede erken biter (summary.yaml: extinct).

Kullanım: uv run python experiments/y_run.py --cond mu1 --seed 0 [--config configs/life_world.yaml --set steps=2000]
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nca.life_world import LifeParams, LifeWorld  # noqa: E402
from nca.train_life import load_life, render  # noqa: E402
from nca.utils import ROOT, load_config, save_grid, to_uint8  # noqa: E402


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/life_world.yaml")
    ap.add_argument("--cond", choices=["mu1", "mu0"], default="mu1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="lifeworld")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config, args.set)
    model, phys = load_life(ROOT / "runs" / cfg["run"])
    model.film_scale = cfg["film_scale"]
    wp = dict(cfg["world"])
    if args.cond == "mu0":
        wp["mu"] = 0.0
    out = ROOT / "runs" / f"{args.name}_{args.cond}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "cond": args.cond, "seed": args.seed}, allow_unicode=True))
    w = LifeWorld(model, phys, size=cfg["size"], n_seeds=cfg["n_seeds"], params=LifeParams(**wp), seed=args.seed)
    t0 = time.time()
    with torch.no_grad():
        while w.t < cfg["steps"] and not w.extinct:
            w.step()
            if w.t % cfg["snapshot_every"] == 0:
                save_grid(out / f"snap_{w.t:06d}.png", list(to_uint8(render(w.x))), scale=2)
                h = w.history[-1]
                print(f"[{out.name}] t={w.t} birey={h['n_org']} hız={h['speed']:.3f} avcılık={h['pred_share']:.3f} "
                      f"kesme={h['cut_rate']:.5f} toplam_birey={len(w.orgs)} {time.time() - t0:.0f}s", flush=True)
    write_csv(out / "history.csv", w.history)
    write_csv(out / "events.csv", [dict(zip(["t", "type", "id", "parent", "cause"], e)) for e in w.events])
    (out / "orgs.json").write_text(json.dumps([asdict(o) for o in w.orgs.values()], default=float))
    (out / "summary.yaml").write_text(yaml.safe_dump({
        "t": w.t, "extinct": bool(w.extinct), "n_orgs_total": len(w.orgs), "sec": round(time.time() - t0),
        "params": asdict(w.p), "film_scale": model.film_scale}, allow_unicode=True))
    print(f"[{out.name}] bitti t={w.t} tükenme={w.extinct} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
