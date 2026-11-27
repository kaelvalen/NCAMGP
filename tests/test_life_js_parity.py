"""demo/life.js'nin nca/life.py + nca/life_world.py ile aynı dünyayı hesapladığını doğrular (node gerekir)."""

import importlib.util
import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from nca.life import E_CH, M_CH, LifeCA, LifePhysics
from nca.life_world import LifeParams, LifeWorld
from nca.utils import ROOT

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node yok")
_spec = importlib.util.spec_from_file_location("export_life", ROOT / "export/export_life.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)
WORLD = {"mu": 0.0, "mu_base": 0.0, "cut_rate": 0.0, "min_viable": 10}


def run_js(tmp_path, script, data):
    (tmp_path / "in.json").write_text(json.dumps(data))
    (tmp_path / "run.js").write_text(script)
    out = subprocess.run([NODE, str(tmp_path / "run.js"), str(ROOT / "demo" / "life.js"), str(tmp_path / "in.json")],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def three_bodies(tmp_path, size=48):
    """Sarmal dünyada 3 beden: biri sol-sağ kenardan geçen, ikisi birbirine değen (yeme ve engel)."""
    torch.manual_seed(0)
    m = LifeCA()
    torch.nn.init.normal_(m.w2.weight, std=0.1)
    m.film_scale, m.fire_rate = 0.5, 1.0
    from dataclasses import asdict
    torch.save({"state_dict": m.state_dict(), "model_config": m.config(), "physics": asdict(LifePhysics(torus=True))},
               tmp_path / "model.pt")
    w = LifeWorld(m, LifePhysics(torus=True), size=size, n_seeds=0, params=LifeParams(**WORLD), seed=0)
    for (ys, xs), tv in [((slice(10, 17), slice(10, 18)), 3.0), ((slice(10, 17), slice(18, 25)), 7.0),
                         ((slice(30, 37), slice(44, 48)), 5.0), ((slice(30, 37), slice(0, 4)), 5.0)]:
        w.x[0, :, ys, xs] = torch.rand(m.channels, ys.stop - ys.start, xs.stop - xs.start)
        w.x[0, M_CH, ys, xs] = 1.0
        w.x[0, E_CH, ys, xs] = 1.0
        w.tag[0, 0, ys, xs] = tv
        w.g[0, :, ys, xs] = torch.randn(m.n_genome, 1, 1) * (tv / 5)
    model = ex.export(tmp_path, {"film_scale": 0.5, "world": WORLD})
    model["fire_rate"] = 1.0
    return w, model


def hwc(t):
    return t[0].permute(1, 2, 0).flatten().tolist()


STEP_JS = """
const { LifeWorld } = require(process.argv[2]);
const d = JSON.parse(require('fs').readFileSync(process.argv[3]));
const w = new LifeWorld(d.model, d.H, d.W);
w.x.set(d.x); w.g.set(d.g); w.R.set(d.R); w.tag.set(d.tag);
for (let i = 0; i < d.steps; i++) w.step();
process.stdout.write(JSON.stringify({x: Array.from(w.x), g: Array.from(w.g), R: Array.from(w.R), tag: Array.from(w.tag)}));
"""


def test_life_js_matches_python(tmp_path):
    w, model = three_bodies(tmp_path)
    data = {"model": model, "H": w.size, "W": w.size, "steps": 5,
            "x": hwc(w.x), "g": hwc(w.g), "R": hwc(w.R), "tag": hwc(w.tag)}
    for _ in range(5):
        w.step()
    js = run_js(tmp_path, STEP_JS, data)
    for k in ("x", "g", "R", "tag"):
        py = getattr(w, k)[0].permute(1, 2, 0).numpy().ravel()
        assert np.abs(np.array(js[k], np.float32) - py).max() < 1e-4, k


CONSERVE_JS = """
const { LifeWorld } = require(process.argv[2]);
const d = JSON.parse(require('fs').readFileSync(process.argv[3]));
const w = new LifeWorld(d.model, d.state.H, d.state.W, {mu: 0.05, mu_base: 0.03, fire_rate: 0.5});
w.loadState(d.state);
const n0 = w.stats().nutrient;
for (let i = 0; i < d.steps; i++) w.step();
process.stdout.write(JSON.stringify({n0, n1: w.stats().nutrient, t: w.t, n_org: w.stats().n_org}));
"""


def test_life_js_conserves_nutrient(tmp_path):
    w, model = three_bodies(tmp_path)
    np.savez(tmp_path / "state_000000.npz", t=0, tag=w.tag[0, 0].numpy(), x=w.x[0].numpy(), g=w.g[0].numpy(),
             R=w.R[0, 0].numpy())
    r = run_js(tmp_path, CONSERVE_JS, {"model": model, "state": ex.pack_state(tmp_path / "state_000000.npz"),
                                       "steps": 2000})
    assert r["t"] == 2000
    assert abs(r["n1"] - r["n0"]) / r["n0"] < 1e-4


TRACK_JS = """
const { LifeWorld } = require(process.argv[2]);
const d = JSON.parse(require('fs').readFileSync(process.argv[3]));
const w = new LifeWorld(d.model, 32, 32);
const C = w.C, M = 3, E = 5, at = (y, x) => y * 32 + x;
const body = (y0, y1, x0, x1, v) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
  w.x[at(y, x) * C + M] = v; w.x[at(y, x) * C + E] = v; w.tag[at(y, x)] = v ? 3.0 : 0; } };
body(10, 16, 4, 20, 1);
w.track();
body(10, 16, 15, 17, 0);  // 6x3 = 18 hücrelik tomurcuk koptu (< min_org_area 20)
w.t += w.p.track_every; w.retagSplits(); w.track();
const bud = [...w.orgs.values()].filter(o => o.area === 18);
const r = {n_bud: bud.length, parent: bud[0].parent, n_org: w.stats().n_org};
// tomurcuk örtüşmesiz kayar: soy etiketten bulunur
const tv = w.tag[at(10, 17)];
for (let y = 10; y < 16; y++) for (let x = 17; x < 20; x++) {
  const s = at(y, x), dd = at(y + 10, x);
  for (let c = 0; c < C; c++) { w.x[dd * C + c] = w.x[s * C + c]; w.x[s * C + c] = 0; }
  w.tag[dd] = w.tag[s]; w.tag[s] = 0; }
w.t += w.p.track_every; w.track();
const o = w.orgs.get(bud[0].id);
r.alive = o.alive; r.area = o.area; r.tag_changed = tv !== 3.0;
process.stdout.write(JSON.stringify(r));
"""


def test_life_js_track(tmp_path):
    _, model = three_bodies(tmp_path)
    r = run_js(tmp_path, TRACK_JS, {"model": model})
    assert r == {"n_bud": 1, "parent": 0, "n_org": 1, "alive": True, "area": 18, "tag_changed": True}
