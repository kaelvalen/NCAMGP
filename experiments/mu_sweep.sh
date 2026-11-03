#!/usr/bin/env bash
# μ taraması: her μ için 2 tohum x 30k adım evrim + uygunluk eğrisi (E5).
set -eu
cd "$(dirname "$0")"
for mu in 0.05 0.5 1.0; do
  uv run python e3_evolution.py --run ancestor_s0 --steps 30000 --seeds 0 1 --no-competition --only-evo \
    --mu $mu --cond-name evrim_mu$mu --history e3_history_mu$mu.csv
done
uv run python e5_fitness.py curve --k 8 --reps 2 --steps 4000 --only evrim_mu0.05 evrim_mu0.5 evrim_mu1.0 --out e5_curve_mu.csv
