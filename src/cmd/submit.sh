#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_SCRIPT="$ROOT_DIR/src/bench_job_cpu.sh"
JOB_NAME="bench_cpu_node"

# Lire variables d'environnement par défaut
BENCH_DURATION=${BENCH_DURATION:-2.0}
BENCH_REPEATS=${BENCH_REPEATS:-3}
BENCH_VERBOSE=${BENCH_VERBOSE:-0}
INCLUDE_NODES=${INCLUDE_NODES:-}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
LIMIT_NODES=${LIMIT_NODES:-}
ONLY_NEW=${ONLY_NEW:-0}

check_deps submit

# build préalable
"$SCRIPT_DIR/build.sh"

echo "[submit] Détection des nœuds idle…"
mapfile -t NODES < <(idle_nodes)
if [[ ${#NODES[@]} -eq 0 ]]; then
  echo "Aucun nœud idle trouvé." >&2
  exit 1
fi
echo "[submit] ${#NODES[@]} nœud(s) idle: ${NODES[*]}"

# include
if [[ -n "$INCLUDE_NODES" ]]; then
  IFS=',' read -r -a inc <<<"$INCLUDE_NODES"
  tmp=()
  for n in "${NODES[@]}"; do
    for i in "${inc[@]}"; do [[ "$n" == "$i" ]] && tmp+=("$n"); done
  done
  NODES=("${tmp[@]}")
fi

# exclude
if [[ -n "$EXCLUDE_NODES" ]]; then
  IFS=',' read -r -a exc <<<"$EXCLUDE_NODES"
  tmp=()
  for n in "${NODES[@]}"; do
    keep=1; for e in "${exc[@]}"; do [[ "$n" == "$e" ]] && keep=0; done
    (( keep )) && tmp+=("$n")
  done
  NODES=("${tmp[@]}")
fi

# limit
if [[ -n "$LIMIT_NODES" ]]; then
  if [[ "$LIMIT_NODES" =~ ^[0-9]+$ ]]; then
    NODES=("${NODES[@]:0:LIMIT_NODES}")
  else
    echo "--limit attend un entier." >&2; exit 1
  fi
fi

# only new
if (( ONLY_NEW )); then
  tmp=()
  for n in "${NODES[@]}"; do
    f="$RES_DIR/$n.csv"
    if [[ ! -f "$f" ]]; then
      tmp+=("$n")
    else
      if awk -F, 'FNR==1{next} {c++} END{exit !(c==0)}' "$f" ; then
        tmp+=("$n")
      fi
    fi
  done
  NODES=("${tmp[@]}")
fi

if [[ ${#NODES[@]} -eq 0 ]]; then
  echo "[submit] Aucun nœud à soumettre après filtres." >&2
  exit 0
fi

wall_s=$(estimate_walltime)
wall=$(fmt_hms "$wall_s")
echo "[submit] Walltime estimé: $wall (sec=$wall_s)"

for NODE in "${NODES[@]}"; do
  if sinfo -h -N -t idle -o '%N' | grep -qx "$NODE"; then
    echo "[submit] Soumission sur $NODE"
    CPUS_NODE=$(sinfo -h -n "$NODE" -o '%c' | tr -d ' ')
    sb_cmd=( sbatch
      --job-name "$JOB_NAME"
      --nodelist "$NODE"
      --nodes 1
      --ntasks-per-node 1
      --cpus-per-task "$CPUS_NODE"
      --exclusive
      --mem=0
      --time "$wall"
      --output "$OUT_DIR/bench_%N.out"
      --error "$OUT_DIR/bench_%N.err"
      --export "ALL,BENCH_ROOT=$ROOT_DIR,BENCH_DURATION=$BENCH_DURATION,BENCH_REPEATS=$BENCH_REPEATS,BENCH_VERBOSE=$BENCH_VERBOSE"
      "$JOB_SCRIPT" )

    if (( BENCH_VERBOSE == 1 )); then
      printf '[submit] CMD: '
      printf '%q ' "${sb_cmd[@]}"
      echo
    fi
    "${sb_cmd[@]}"
  else
    echo "[submit] $NODE n'est plus idle, on saute."
  fi
done

echo "[submit] Soumissions terminées."
