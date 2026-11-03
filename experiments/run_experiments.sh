#!/usr/bin/env bash
# Eğitilmiş modellerle D1-D6 deneylerini, analiz şekillerini ve demo dışa aktarımını çalıştırır.
# Kullanım: bash experiments/run_experiments.sh [--quick]
set -eu
cd "$(dirname "$0")/.."
N=100; [ "${1:-}" = "--quick" ] && N=16
PY="uv run python"
$PY experiments/d1_reproduce.py --n $N
$PY experiments/d2_fragment_curve.py --mode inplace --n $N
$PY experiments/d2_fragment_curve.py --mode split --n $N
$PY experiments/d3_cut_direction.py --n $N
$PY experiments/d4_probe.py --runs 'r?_s*'
$PY experiments/d5_intervention.py --runs 'r?_s*' --n $N
if ls runs/abl_*/model.pt >/dev/null 2>&1; then
  $PY experiments/d1_reproduce.py --runs 'abl_*' 'r3_s0' --n $N --out ablation_d1.csv
  $PY experiments/d2_fragment_curve.py --mode inplace --runs 'abl_*' 'r3_s0' --n $N --out ablation_d2.csv
fi
if ls runs/noise_*/model.pt >/dev/null 2>&1; then $PY experiments/d6_noise.py --n $N; fi
$PY analysis/plots.py
$PY export/export_weights.py
