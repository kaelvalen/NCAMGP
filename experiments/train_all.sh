#!/usr/bin/env bash
# Tüm modelleri sırayla eğitir; model.pt'si olan koşuları atlar.
# Kullanım: bash experiments/train_all.sh [core|ablations|all]
set -u
cd "$(dirname "$0")/.."
WHAT=${1:-all}
train() {  # config seed
  local name; name=$(grep '^name:' "$1" | awk '{print $2}')
  if [ -f "runs/${name}_s$2/model.pt" ]; then echo "atla: ${name}_s$2"; return; fi
  uv run python -m nca.train "$1" --seed "$2" || echo "HATA: $1 seed=$2"
}
if [ "$WHAT" = core ] || [ "$WHAT" = all ]; then
  for s in 0 1 2; do for r in 0 1 2 3; do train configs/r$r.yaml $s; done; done
fi
if [ "$WHAT" = ablations ] || [ "$WHAT" = all ]; then
  for s in ${ABL_SEEDS:-0}; do for c in configs/ablations/*.yaml; do train "$c" $s; done; done
fi
