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

check_deps submit

# Fonction issue de src/list_free_gpu.sh: liste les nœuds avec GPU tous libres
list_free_gpu_nodes() {
  # On boucle sur tous les nœuds connus de Slurm
  for node in $(sinfo -h -N -o "%N"); do
    # Vérifie si le nœud a des GPU configurés
    if scontrol show node "$node" | grep -q "Gres=gpu"; then
      # Récupère nombre total et nombre utilisés
      total=$(scontrol show node "$node" | awk -F= '/CfgTRES/{print $2}' | grep -o "gres/gpu=[0-9]*" | cut -d= -f2)
      used=$(scontrol show node "$node" | awk -F= '/AllocTRES/{print $2}' | grep -o "gres/gpu=[0-9]*" | cut -d= -f2)
      # Si aucun GPU n'est utilisé et qu'il y en a au moins 1 → on affiche le nœud
      if [[ -n "$total" && "$total" -gt 0 && ( -z "$used" || "$used" -eq 0 ) ]]; then
        echo "$node"
      fi
    fi
  done
}

# Détecter les nœuds avec tous les GPU libres
mapfile -t GPU_NODES < <(list_free_gpu_nodes)
if (( BENCH_VERBOSE == 1 )); then
  echo "[submit-gpu] Nœuds GPU libres: ${GPU_NODES[*]:-none}"
fi

if [[ ${#GPU_NODES[@]} -eq 0 ]]; then
  echo "Aucun nœud GPU libre trouvé." >&2
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

wall_s=$(estimate_walltime)
wall=$(fmt_hms "$wall_s")
echo "[submit-gpu] Walltime estimé: $wall (sec=$wall_s)"

for NODE in "${GPU_NODES[@]}"; do
  # Vérifier encore dispo/idle
  if scontrol show node "$NODE" | grep -q "Gres=gpu"; then
    echo "[submit-gpu] Soumission sur $NODE"
    sb_cmd=( sbatch
      --job-name "$JOB_NAME"
      --nodelist "$NODE"
      --nodes 1
      --ntasks-per-node 1
      --gres=gpu:1
      --exclusive
      --mem=0
      --time "$wall"
      --output "$OUT_DIR/bench_%N.out"
      --error "$OUT_DIR/bench_%N.err"
      --export "ALL,BENCH_ROOT=$ROOT_DIR,BENCH_DURATION=$BENCH_DURATION,BENCH_REPEATS=$BENCH_REPEATS,BENCH_VERBOSE=$BENCH_VERBOSE"
      "$JOB_SCRIPT" )

    if (( BENCH_VERBOSE == 1 )); then
      printf '[submit-gpu] CMD: '
      printf '%q ' "${sb_cmd[@]}"
      echo
    fi
    "${sb_cmd[@]}"
  else
    echo "[submit-gpu] $NODE n'est plus disponible, on saute."
  fi
done

echo "[submit-gpu] Soumissions terminées."
