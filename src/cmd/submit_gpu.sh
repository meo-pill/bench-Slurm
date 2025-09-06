#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

JOB_SCRIPT="$ROOT_DIR/src/bench_job_gpu.sh"
JOB_NAME="bench_gpu_node"

# Lire variables d'environnement par défaut
BENCH_DURATION=${BENCH_DURATION:-2.0}
BENCH_REPEATS=${BENCH_REPEATS:-3}
BENCH_VERBOSE=${BENCH_VERBOSE:-0}
INCLUDE_NODES=${INCLUDE_NODES:-}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
LIMIT_NODES=${LIMIT_NODES:-}
GPU_WALLTIME_FACTOR=${GPU_WALLTIME_FACTOR:-10}  # facteur multiplicatif pour walltime GPU

check_deps submit

# build préalable
"$SCRIPT_DIR/build.sh"

# Fonction issue de src/list_free_gpu.sh: liste les nœuds avec GPU tous libres
get_busy_gpu_nodes() {
  # Liste les hôtes qui ont au moins un GPU alloué par des jobs RUNNING
  squeue -h -o "%R %.b" --states=RUNNING \
    | awk '$2 ~ /gpu/' \
    | cut -d' ' -f1 \
    | while read -r nodes; do scontrol show hostnames "$nodes"; done \
    | sort -u
}

BUSY_GPU_NODES=$(get_busy_gpu_nodes || true)
is_busy_node() { grep -qxF "$1" <<<"$BUSY_GPU_NODES"; }

# Dresse la liste des nœuds avec GPU et leur quantité (une seule fois)
nodes_with_gpu_counts() {
  for node in $(sinfo -h -N -o "%N"); do
    line=$(scontrol show node -o "$node" 2>/dev/null || true)
    gres=$(sed -n 's/.*Gres=\([^ ]*\).*/\1/p' <<<"$line")
    total=0
    if [[ -n "$gres" && "$gres" != "(null)" ]]; then
      total=$(tr ',' '\n' <<<"$gres" | awk -F: '
        $1=="gpu" { n=$NF; gsub(/[^0-9].*/, "", n); s+=n }
        END{print s+0}
      ')
    fi
    if [[ "$total" -gt 0 ]]; then
      echo "$node $total"
    fi
  done
}

# Construire la table "nœud -> nombre de GPU" une fois
declare -A GPU_COUNT
GPU_NODES=()
while read -r _n _c; do
  GPU_COUNT["$_n"]="$_c"
  GPU_NODES+=("$_n")
done < <(nodes_with_gpu_counts)

if (( BENCH_VERBOSE == 1 )); then
  echo "[submit-gpu] Nœuds avec GPU détectés: ${GPU_NODES[*]:-none}"
  for n in "${GPU_NODES[@]}"; do echo "  - $n: ${GPU_COUNT[$n]}"; done
fi

if [[ ${#GPU_NODES[@]} -eq 0 ]]; then
  echo "Aucun nœud avec GPU détecté." >&2
  exit 1
fi

# include
if [[ -n "$INCLUDE_NODES" ]]; then
  IFS=',' read -r -a inc <<<"$INCLUDE_NODES"
  tmp=()
  for n in "${GPU_NODES[@]}"; do for i in "${inc[@]}"; do [[ "$n" == "$i" ]] && tmp+=("$n"); done; done
  GPU_NODES=("${tmp[@]}")
fi

# exclude
if [[ -n "$EXCLUDE_NODES" ]]; then
  IFS=',' read -r -a exc <<<"$EXCLUDE_NODES"
  tmp=()
  for n in "${GPU_NODES[@]}"; do keep=1; for e in "${exc[@]}"; do [[ "$n" == "$e" ]] && keep=0; done; (( keep )) && tmp+=("$n"); done
  GPU_NODES=("${tmp[@]}")
fi

# limit
if [[ -n "$LIMIT_NODES" ]]; then
  if [[ "$LIMIT_NODES" =~ ^[0-9]+$ ]]; then
    GPU_NODES=("${GPU_NODES[@]:0:LIMIT_NODES}")
  else
    echo "--limit attend un entier." >&2; exit 1
  fi
fi

if [[ ${#GPU_NODES[@]} -eq 0 ]]; then
  echo "[submit-gpu] Aucun nœud GPU à soumettre après filtres." >&2
  exit 0
fi

wall_s="$(estimate_walltime)*$GPU_WALLTIME_FACTOR"
wall=$(fmt_hms "$wall_s")
echo "[submit-gpu] Walltime estimé: $wall (sec=$wall_s)"

BUSY_GPU_NODES=$(get_busy_gpu_nodes || true)
for NODE in "${GPU_NODES[@]}"; do
  # Vérifier au dernier moment si le nœud a des GPU occupés
  if is_busy_node "$NODE"; then
    (( BENCH_VERBOSE == 1 )) && echo "[submit-gpu] $NODE occupé (GPU en usage) — on saute."
    continue
  fi
  TOTAL_GPU=${GPU_COUNT[$NODE]:-0}
  echo "[submit-gpu] Soumission sur $NODE (GPU=$TOTAL_GPU)"
  sb_cmd=( sbatch
      --job-name "$JOB_NAME"
      --nodelist "$NODE"
      --nodes 1
      --ntasks-per-node 1
      --cpus-per-task 8
      --gres=gpu:"$TOTAL_GPU"
      --mem=20G
      --time "$wall"
      --output "$OUT_DIR/bench_%N_gpu.out"
      --error "$OUT_DIR/bench_%N_gpu.err"
      --export "ALL,BENCH_ROOT=$ROOT_DIR,BENCH_DURATION=$BENCH_DURATION,BENCH_REPEATS=$BENCH_REPEATS,BENCH_VERBOSE=$BENCH_VERBOSE,BENCH_VRAM_FRAC=${BENCH_VRAM_FRAC:-}"
      "$JOB_SCRIPT" )

    if (( BENCH_VERBOSE == 1 )); then
      printf '[submit-gpu] CMD: '
      printf '%q ' "${sb_cmd[@]}"
      echo
    fi
    "${sb_cmd[@]}"
done

echo "[submit-gpu] Soumissions terminées."
