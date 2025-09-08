#!/bin/bash
# Slurm job script GPU: exécute le bench GPU (mono+multi, tous backends) sur le nœud local
# Prérequis: BENCH_ROOT exporté par sbatch (comme pour bench_job_cpu.sh)

set -euo pipefail

ROOT_DIR=${BENCH_ROOT:?BENCH_ROOT non défini}
OUT_DIR="$ROOT_DIR/outputs"
SRC_DIR="$ROOT_DIR/src"
RES_DIR="$ROOT_DIR/results"
PY=${BENCH_PYTHON:-python3}

enforce_conda_presence() {
    # Exige que 'conda' soit disponible et qu'un environnement soit activé.
    if ! command -v conda >/dev/null 2>&1; then
        echo "[gpu] ERREUR: conda est requis pour l'exécution GPU (binaire 'conda' introuvable)." >&2
        exit 1
    fi
    if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
        echo "[gpu] ERREUR: un environnement conda actif est requis. Ex: 'conda activate bench'." >&2
        exit 1
    fi
    if [[ -n "${BENCH_CONDA_ENV:-}" ]]; then
        local cur
        cur="${CONDA_DEFAULT_ENV:-$(basename "${CONDA_PREFIX:-}")}" || cur="(inconnu)"
        if [[ "$cur" != "$BENCH_CONDA_ENV" ]]; then
            echo "[gpu] ERREUR: environnement conda actif '$cur' différent de l'environnement requis '$BENCH_CONDA_ENV'." >&2
            exit 1
        fi
    fi
}

enforce_conda_presence

HOST=$(hostname -s)
DUR=3.0
REPEATS=5
VERBOSE=0
WARMUP_ARG=""
VRAM_FRAC_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
        DUR="${2:?valeur manquante pour --duration}"; shift 2 ;;
        --repeats)
        REPEATS="${2:?valeur manquante pour --repeats}"; shift 2 ;;
        --verbose)
        VERBOSE=1; shift ;;
        --warmup)
        WARMUP_ARG="${2:?valeur manquante pour --warmup}"; shift 2 ;;
        --vram-frac)
        VRAM_FRAC_ARG="${2:?valeur manquante pour --vram-frac}"; shift 2 ;;
        --)
        shift; break ;;
        *)
        echo "[bench_job_gpu] option inconnue: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR" "$RES_DIR"

# Vérifie python3 et qu'au moins un backend GPU est disponible (torch/cupy/numba)
check_deps() {
    local missing=()
    command -v "$PY" >/dev/null 2>&1 || missing+=("python")
    if (( ${#missing[@]} > 0 )); then
        echo "Dépendances manquantes: ${missing[*]}" >&2
        # Ne marque pas le job en échec
        exit 0
    fi
  if ! "$PY" - <<'PY'
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
sys.exit(0 if ok else 2)
PY
    then
        echo "Aucun backend GPU disponible (torch/cupy/numba) sur ce nœud." >&2
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

# Commande bench GPU (écrit une seule ligne dans results/gpu_<node>.csv)
CMD=("$PY" "$SRC_DIR/gpu_bench.py" --duration "$DUR" --repeats "$REPEATS" --node "$HOST" --csv-dir "$RES_DIR")
if [[ -n "${BENCH_CONDA_ENV:-}" ]]; then
    CMD+=(--conda-env "$BENCH_CONDA_ENV")
fi
[[ -n "$WARMUP_ARG" ]] && CMD+=(--warmup "$WARMUP_ARG")
[[ -n "$VRAM_FRAC_ARG" ]] && CMD+=(--vram-frac "$VRAM_FRAC_ARG")
(( VERBOSE == 1 )) && CMD+=(--verbose)

# Lancer en laissant stderr aller au .err Slurm; ne pas faire échouer le job
set +e
"${CMD[@]}" 2> >(tee >&2)
rc=$?
set -e

if (( rc != 0 )); then
    echo "[gpu] ÉCHEC: exécution bench (rc=$rc). Job Slurm marqué en erreur." >&2
    exit $rc
fi

csv="$RES_DIR/gpu_${HOST}.csv"
if [[ -f "$csv" ]]; then
    echo "[gpu] Bench terminé. Ligne ajoutée dans: $csv"
else
    echo "[gpu] Bench terminé, mais fichier CSV introuvable: $csv" >&2
fi
