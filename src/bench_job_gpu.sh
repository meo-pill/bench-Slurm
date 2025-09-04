#!/bin/bash
# Slurm job script GPU: exécute le bench GPU (mono+multi, tous backends) sur le nœud local
# Prérequis: BENCH_ROOT exporté par sbatch (comme pour bench_job_cpu.sh)

set -euo pipefail

ROOT_DIR=${BENCH_ROOT:?BENCH_ROOT non défini}
OUT_DIR="$ROOT_DIR/outputs"
SRC_DIR="$ROOT_DIR/src"

HOST=$(hostname -s)
DUR=${BENCH_DURATION:-2.0}
REPEATS=${BENCH_REPEATS:-3}
VERBOSE=${BENCH_VERBOSE:-0}

mkdir -p "$OUT_DIR"

# Vérifie python3 et qu'au moins un backend GPU est disponible (torch/cupy/numba/pyopencl)
check_deps() {
  local missing=()
  command -v python3 >/dev/null 2>&1 || missing+=("python3")
  if (( ${#missing[@]} > 0 )); then
  echo "Dépendances manquantes: ${missing[*]}" >&2
  # Ne marque pas le job en échec
  exit 0
  fi
  if ! python3 - <<'PY'
import sys
ok = False
try:
  import torch
  if getattr(torch, "cuda", None) and torch.cuda.is_available():
    ok = True
except Exception:
  pass
try:
  import cupy as cp
  try:
    n = cp.cuda.runtime.getDeviceCount()
    if n and n > 0:
      ok = True
  except Exception:
    pass
except Exception:
  pass
try:
  from numba import cuda as _cuda
  try:
    if _cuda.is_available():
      ok = True
  except Exception:
    pass
except Exception:
  pass
try:
  import pyopencl as cl
  has_gpu = False
  try:
    for p in cl.get_platforms():
      devs = p.get_devices(device_type=cl.device_type.GPU)
      if devs:
        has_gpu = True
        break
  except Exception:
    has_gpu = False
  if has_gpu:
    ok = True
except Exception:
  pass
sys.exit(0 if ok else 2)
PY
  then
  echo "Aucun backend GPU disponible (torch/cupy/numba/pyopencl) sur ce nœud." >&2
  exit 0
  fi
}

check_deps

# Empêcher une exécution concurrente GPU sur le même nœud (si FS partagé)
lockfile="$OUT_DIR/.lock.gpu.$HOST"
if ! ( set -o noclobber; : >"$lockfile" ) 2>/dev/null; then
  echo "Un bench GPU est déjà en cours pour $HOST, on quitte." >&2
  exit 0
fi
trap 'rm -f "$lockfile"' EXIT

# Commande bench GPU (écrit une seule ligne dans outputs/gpu_<node>.csv)
CMD=(python3 "$SRC_DIR/gpu_bench.py" --duration "$DUR" --repeats "$REPEATS" --node "$HOST")
(( VERBOSE == 1 )) && CMD+=(--verbose)

# Lancer en laissant stderr aller au .err Slurm; ne pas faire échouer le job
set +e
"${CMD[@]}" 2> >(tee >&2)
rc=$?
set -e

if (( rc != 0 )); then
  echo "[gpu] échec de l'exécution (rc=$rc). Voir le .err pour les détails." >&2
  # Ne pas marquer le job en échec pour rester cohérent avec bench CPU (on sort 0)
  exit 0
fi

csv="$OUT_DIR/gpu_${HOST}.csv"
if [[ -f "$csv" ]]; then
  echo "[gpu] Bench terminé. Ligne ajoutée dans: $csv"
else
  echo "[gpu] Bench terminé, mais fichier CSV introuvable: $csv" >&2
fi
