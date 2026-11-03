"""demo/nca.js'nin nca/model.py ile aynı adımı hesapladığını doğrular (node gerekir)."""

import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from nca.model import PERCEPTIONS, CAModel
from nca.utils import ROOT, load_target, pad_to

NODE = shutil.which("node")

SCRIPT = """
const { NCA } = require(process.argv[2]);
const d = JSON.parse(require('fs').readFileSync(process.argv[3]));
const nca = new NCA(d.model, d.H, d.W);
nca.state.set(d.state);
const fire = new Uint8Array(d.H * d.W).fill(1);
for (let i = 0; i < d.steps; i++) nca.step(fire);
process.stdout.write(JSON.stringify(Array.from(nca.state)));
"""


@pytest.mark.skipif(NODE is None, reason="node yok")
@pytest.mark.parametrize("perception", ["sobel_lap", "sobel", "lap"])
def test_js_matches_python(tmp_path, perception):
    torch.manual_seed(0)
    m = CAModel(perception=perception)
    torch.nn.init.normal_(m.w2.weight, std=0.02)
    size, steps = 48, 5
    x = torch.zeros(1, m.channels, size, size)
    x[0, :4] = pad_to(load_target()["rgba"], size)
    x[0, 4:] = torch.rand(m.n_hidden, size, size) * x[0, 3:4]
    with torch.no_grad():
        y = m.rollout(x, steps, fire_rate=1.0)
    model = {
        "channels": m.channels,
        "hidden_dim": m.hidden_dim,
        "filters": PERCEPTIONS[perception],
        "fire_rate": 1.0,
        "w1": m.w1.weight[:, :, 0, 0].detach().flatten().tolist(),
        "b1": m.w1.bias.detach().tolist(),
        "w2": m.w2.weight[:, :, 0, 0].detach().flatten().tolist(),
    }
    data = {"model": model, "H": size, "W": size, "steps": steps, "state": x[0].permute(1, 2, 0).flatten().tolist()}
    (tmp_path / "in.json").write_text(json.dumps(data))
    (tmp_path / "run.js").write_text(SCRIPT)
    out = subprocess.run(
        [NODE, str(tmp_path / "run.js"), str(ROOT / "demo" / "nca.js"), str(tmp_path / "in.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    js = np.array(json.loads(out), np.float32).reshape(size, size, m.channels)
    py = y[0].permute(1, 2, 0).numpy()
    assert np.abs(js - py).max() < 1e-4
