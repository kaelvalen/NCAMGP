"""Evrimsel fizik: kalıtım, potansiyel, kesme ve JS/Python paritesi."""

import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from nca.evo import V_CH, EvoCA, Physics, alive_cells, inherit, make_seed, step, wound
from nca.model import PERCEPTIONS
from nca.utils import ROOT

NODE = shutil.which("node")


def test_genome_inherited_by_new_cells():
    m = EvoCA()
    # sabit büyüme kuralı: her güncellemede alfa artar (kalıtımı kuraldan bağımsız test eder)
    m.update = lambda x, g: torch.cat([torch.zeros_like(x[:, :3]), torch.full_like(x[:, 3:4], 0.2), torch.zeros_like(x[:, 4:])], 1)
    gv = torch.randn(1, 8)
    x, g = make_seed(gv, 32, m.channels)
    for _ in range(5):  # sabit kural her yere yayılır; ızgara dolmadan dur
        x, g = step(m, x, g, Physics(), fire_rate=1.0)
    alive = alive_cells(x)[0, 0]
    assert 1 < alive.sum() < alive.numel()
    assert torch.allclose(g[0][:, alive], gv[0][:, None].expand(-1, int(alive.sum())), atol=1e-5)
    assert g[0][:, ~alive].abs().max() == 0


def test_inherit_born_cell_gets_neighbor_mean():
    g = torch.zeros(1, 2, 3, 3)
    was = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    was[0, 0, 0, 0] = was[0, 0, 0, 2] = True
    g[0, :, 0, 0] = torch.tensor([1.0, 0.0])
    g[0, :, 0, 2] = torch.tensor([3.0, 2.0])
    alive = was.clone()
    alive[0, 0, 0, 1] = True
    out = inherit(g, was, alive, 0.0)
    assert torch.allclose(out[0, :, 0, 1], torch.tensor([2.0, 1.0]))


def test_wound_raises_potential_at_cut_edge():
    m = EvoCA()
    x = torch.zeros(1, m.channels, 16, 16)
    x[0, 3, 4:12, 4:12] = 1
    removed = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    removed[0, 0, 4:12, 8:12] = True
    x[0, :, 4:12, 8:12] = 0
    wound(x, removed, Physics())
    v = x[0, V_CH]
    assert v[6, 7] > 0 and v[6, 4] == 0 and v[6, 9] == 0


@pytest.mark.skipif(NODE is None, reason="node yok")
def test_js_matches_python(tmp_path):
    torch.manual_seed(0)
    m = EvoCA()
    torch.nn.init.normal_(m.w2.weight, std=0.03)
    phys = Physics()
    size, steps = 24, 6
    gv = torch.randn(1, m.n_genome)
    x, g = make_seed(gv, size, m.channels)
    x[0, 3, 8:16, 8:16] = 1
    x[0, 5:, 8:16, 8:16] = torch.rand(m.n_hidden, 8, 8)
    g[0, :, 8:16, 8:16] = gv[0][:, None, None]
    x[0, V_CH, 10:12, 10:14] = 0.5
    x0, g0 = x.clone(), g.clone()
    with torch.no_grad():
        for _ in range(steps):
            x, g = step(m, x, g, phys, fire_rate=1.0)
    model = {
        "channels": m.channels, "n_genome": m.n_genome, "hidden_dim": m.hidden_dim,
        "filters": PERCEPTIONS[m.perception], "fire_rate": 1.0, "sense_v": True, "physics": vars(phys),
        "w1": m.w1.weight[:, :, 0, 0].detach().flatten().tolist(),
        "b1": m.w1.bias.detach().tolist(),
        "w2": m.w2.weight[:, :, 0, 0].detach().flatten().tolist(),
    }
    data = {
        "model": model, "H": size, "W": size, "steps": steps,
        "x": x0[0].permute(1, 2, 0).flatten().tolist(), "g": g0[0].permute(1, 2, 0).flatten().tolist(),
    }
    (tmp_path / "in.json").write_text(json.dumps(data))
    (tmp_path / "run.js").write_text(
        """
const { EvoWorld } = require(process.argv[2]);
const d = JSON.parse(require('fs').readFileSync(process.argv[3]));
const w = new EvoWorld(d.model, d.H, d.W, { energy: false });
w.x.set(d.x); w.g.set(d.g);
const fire = new Uint8Array(d.H * d.W).fill(1);
for (let i = 0; i < d.steps; i++) w.step(fire);
process.stdout.write(JSON.stringify({ x: Array.from(w.x), g: Array.from(w.g) }));
"""
    )
    out = json.loads(
        subprocess.run([NODE, str(tmp_path / "run.js"), str(ROOT / "demo" / "evo.js"), str(tmp_path / "in.json")],
                       capture_output=True, text=True, check=True).stdout
    )
    jx = np.array(out["x"], np.float32).reshape(size, size, m.channels)
    jg = np.array(out["g"], np.float32).reshape(size, size, m.n_genome)
    assert np.abs(jx - x[0].permute(1, 2, 0).numpy()).max() < 1e-4
    assert np.abs(jg - g[0].permute(1, 2, 0).numpy()).max() < 1e-4


def test_mutation_scale_python():
    from nca.evo import mutate

    x = torch.zeros(1, 17, 64, 64)
    x[:, 3] = 1
    x[:, V_CH] = 1
    g = torch.zeros(1, 8, 64, 64)
    d = mutate(g, x, 1.0)[0, :, 8:-8, 8:-8]
    assert 0.85 < d.std().item() < 1.15


@pytest.mark.skipif(NODE is None, reason="node yok")
def test_mutation_scale_js(tmp_path):
    (tmp_path / "m.js").write_text(
        """
const { EvoWorld } = require(process.argv[2]);
const model = { channels: 17, n_genome: 8, hidden_dim: 4, filters: ['identity'], fire_rate: 1, physics: {},
  w1: new Array(4 * (17 + 8)).fill(0), b1: [0, 0, 0, 0], w2: new Array(17 * 4).fill(0) };
const w = new EvoWorld(model, 64, 64, { mu: 1, g_clip: 100 });
for (let i = 0; i < 64 * 64; i++) { w.x[i * 17 + 3] = 1; w.x[i * 17 + 4] = 1; }
w.mutate(Math.random);
let s = 0, n = 0;
for (let y = 8; y < 56; y++) for (let x = 8; x < 56; x++) for (let q = 0; q < 8; q++) { const v = w.g[(y * 64 + x) * 8 + q]; s += v * v; n++; }
process.stdout.write(String(Math.sqrt(s / n)));
"""
    )
    std = float(subprocess.run([NODE, str(tmp_path / "m.js"), str(ROOT / "demo" / "evo.js")], capture_output=True, text=True, check=True).stdout)
    assert 0.85 < std < 1.15


@pytest.mark.skipif(NODE is None, reason="node yok")
def test_cut_drift_js(tmp_path):
    """Kesilen parça adım adım ayrılır, enerji korunur; yolu kapalı parça ezilmeden durur."""
    (tmp_path / "d.js").write_text(
        """
const { EvoWorld } = require(process.argv[2]);
const model = { channels: 17, n_genome: 8, hidden_dim: 4, filters: ['identity'], fire_rate: 1, physics: {},
  w1: new Array(4 * (17 + 8)).fill(0), b1: [0, 0, 0, 0], w2: new Array(17 * 4).fill(0) };
const S = 64, w = new EvoWorld(model, S, S, { drift_every: 1 });
const put = (y, x, tag) => { const i = y * S + x; w.x[i * 17 + 3] = 1; w.x[i * 17 + 5] = y + x; w.E[i] = 0.5; w.tag[i] = tag; return i; };
const cells = [];
for (let y = 20; y < 44; y++) for (let x = 28; x < 36; x++) cells.push(put(y, x, 5));
for (let x = 28; x < 36; x++) put(58, x, 9); // alttaki parçanın yolunu kesen başka birey
const E0 = w.E.reduce((a, b) => a + b, 0);
w.R.fill(0);
w.cut(28, 32, 36, 32, null, cells);
const drifts = w.drifts.length, released = w.R[cells[0]];
const span = () => { const t = {}; for (let i = 0; i < S * S; i++) if (w.tag[i] && w.tag[i] !== 9 && w.x[i * 17 + 3] > 0.1) { const y = (i / S) | 0, k = w.tag[i].toFixed(3); t[k] = [Math.min(y, (t[k] || [99])[0]), Math.max(y, (t[k] || [0, 0])[1])]; } return Object.values(t).sort((a, b) => a[0] - b[0]); };
const first = span(), Ecut = w.E.reduce((a, b) => a + b, 0);
for (let k = 0; k < 40; k++) { w.drift(); w.steps++; }
process.stdout.write(JSON.stringify({ drifts, released, first, last: span(), Ecut, E1: w.E.reduce((a, b) => a + b, 0), E0, other: w.x[(58 * S + 30) * 17 + 3] }));
"""
    )
    out = json.loads(subprocess.run([NODE, str(tmp_path / "d.js"), str(ROOT / "demo" / "evo.js")],
                                    capture_output=True, text=True, check=True).stdout)
    assert out["drifts"] == 2 and out["released"] == 1
    assert out["last"][0][1] < out["first"][0][1] - 5  # üst parça uzaklaştı
    assert out["first"][1][0] < out["last"][1][0]  # alt parça uzaklaştı
    assert out["last"][1][1] < 58 and out["other"] == 1  # diğer bireye değmeden durdu
    assert abs(out["E1"] - out["Ecut"]) < 1e-3


CHROME = shutil.which("google-chrome-stable") or shutil.which("google-chrome") or shutil.which("chromium")


@pytest.mark.skipif(CHROME is None, reason="chrome yok")
def test_gpu_rule_matches_js(tmp_path):
    """WebGL2 kuralı, aynı durum ve maskede JS kuralıyla aynı sonucu verir."""
    demo = (ROOT / "demo").as_uri()
    (tmp_path / "t.html").write_text(
        f"""<!doctype html><pre id=o></pre><script src="{demo}/evo_models.js"></script><script src="{demo}/evo.js"></script>
<script src="{demo}/evo_gpu.js"></script><script>
const w = new EvoWorld(EVO_MODELS[0], 128, 128), g = new EvoGPU(w), ok = g.matches(w);
document.getElementById('o').textContent = 'SONUC ' + ok + ' ' + g.error;
</script>"""
    )
    out = subprocess.run([CHROME, "--headless=new", "--enable-unsafe-swiftshader", f"--user-data-dir={tmp_path / 'p'}",
                          "--allow-file-access-from-files", "--dump-dom", (tmp_path / "t.html").as_uri()],
                         capture_output=True, text=True, timeout=120).stdout
    assert "SONUC true" in out, out[-300:]
